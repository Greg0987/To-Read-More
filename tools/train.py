from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Noted: whether to use necessarily
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import time
import os
import gc
from six.moves import cPickle
import traceback
from collections import defaultdict

import sys
sys.path.append(os.getcwd())

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
import skimage.io
import captioning.utils.eval_utils as eval_utils
import captioning.utils.misc as utils
from captioning.modules.rewards import init_scorer
from captioning.modules.loss_wrapper import LossWrapper

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)



def train(opt):
    ##################
    # Build dataloader
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    ##################
    # Initialize infos
    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
    }
    # 加载旧的训练信息，并检查模型是否兼容
    if opt.start_from is not None and os.path.isfile(os.path.join('./outputs', opt.start_from, 'infos_' + opt.id + '.pkl')):
        with open(os.path.join('./outputs', opt.start_from, 'infos_' + opt.id + '.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert getattr(saved_model_opt, checkme) == getattr(opt,
                                                                    checkme), "Command line argument and saved model disagree on '%s' " % checkme
    infos['opt'] = opt

    ##################
    # Build logger
    histories = defaultdict(dict)
    if opt.start_from is not None and os.path.isfile(os.path.join('./outputs', opt.start_from, 'histories_' + opt.id + '.pkl')):
        with open( os.path.join('./outputs', opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
            histories.update(utils.pickle_load(f))
    # tensorboard logger
    tb_summary_writer = SummaryWriter(log_dir=os.path.join('./outputs', opt.checkpoint_path, 'logs'))

    ##################
    # Build model
    opt.vocab = loader.get_vocab()
    model = models.setup(opt)
    # del opt.vocab   # model.vocab = loader.get_vocab()
    # Load pretrained weights: 加载训练过的模型权重
    if opt.start_from is not None and os.path.isfile(os.path.join('./outputs', opt.start_from, 'model.pth')):
        model.load_state_dict(torch.load(os.path.join('./outputs', opt.start_from, 'model.pth')))

    # wrap the model with loss function (criterion) 将损失函数包装到模型输出中
    lw_model = LossWrapper(model, opt)
    # wrap with dataparallel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp_model = torch.nn.DataParallel(model, device_ids=[0]) # 【待修改gpu】
    dp_model.vocab = getattr(model, 'vocab', None)

    if torch.cuda.device_count() > 1:
        dp_lw_model = torch.nn.DataParallel(lw_model)
    else:
        dp_lw_model = torch.nn.DataParallel(lw_model, device_ids=[0])
    dp_lw_model.to(device)

    ##################
    # Bulid optimizer 优化器
    if opt.noamopt:
        assert opt.caption_model in ['transformer', 'bert', 'm2transformer', 'DiM2Transformer', 'tfvae', 'vat'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, optim_func=opt.optim, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer,
                                            factor=opt.reduce_on_plateau_factor,
                                            patience=opt.reduce_on_plateau_patience)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)
    # Load the optimizer    加载优化器
    if opt.start_from is not None and os.path.isfile(os.path.join('./outputs', opt.start_from, "optimizer.pth")):
        pth = torch.load(os.path.join('./outputs', opt.start_from, 'optimizer.pth'))
        if 'optimizer_state_dict' in pth:
            state_dict = pth['optimizer_state_dict']
        else:
            state_dict = pth
        optimizer.load_state_dict(state_dict)

    ##################
    # Get ready to start
    iteration = infos['iter']
    epoch = infos['epoch']
    # For back compatibility
    if 'iterators' in infos:
        infos['loader_state_dict'] = {split: {'index_list': infos['split_ix'][split],
                                              'iter_counter': infos['iterators'][split]}
                                      for split in infos['train', 'val', 'test']}
        # del infos['split_ix']
        # del infos['iterators']
    loader.load_state_dict(infos['loader_state_dict'])
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    if opt.noamopt:
        optimizer._step = iteration
    # flag indicating finish of an epoch
    # Always set to True at the beginning to initialize the lr or etc.
    # 表示完成一个epoch的标志，开始时设置为True，以初始化学习率等参数
    epoch_done = True
    # Assure in training mode   确保处于训练模式
    dp_lw_model.train()
    best_cider = infos.get('best_test_score', 0.0)
    print('The way to reduce the dimension of attention is:', opt.d_reduction)

    # Start training
    try:
        while True:
            # Stop if reaching max epochs
            if epoch >= opt.max_epochs and opt.max_epochs != -1:
                break
            # Update the iteration and epoch    # 一轮后，更新参数
            if epoch_done:
                if not opt.noamopt and not opt.reduce_on_plateau:
                    # Assign the learning rate
                    if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                        frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                        decay_factor = opt.learning_rate_decay_rate ** frac
                        opt.current_lr = opt.learning_rate * decay_factor
                    else:
                        opt.current_lr = opt.learning_rate
                    utils.set_lr(optimizer, opt.current_lr) # set the decayed rate  设置衰减率
                # Assign the scheduled sampling prob  分配计划抽样概率
                if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                    frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                    opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                    model.ss_prob = opt.ss_prob
                # If start self-critical training    开始自批判训练
                if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                    SCST_flag = True
                    init_scorer(opt.cached_tokens)
                else:
                    SCST_flag = False
                epoch_done = False

            start = time.time()
            if opt.use_warmup and (iteration <= opt.noamopt_warmup):    # 使用热身
                opt.current_lr = opt.learning_rate * (iteration + 1) / opt.noamopt_warmup
                utils.set_lr(optimizer, opt.current_lr)
            # Load data from train split (0) or validation split (1)    从训练集（0）或验证集（1）加载数据
            data = loader.get_batch('train')
            print("=" * 80)
            print('Read data:', time.time() - start)
            torch.cuda.synchronize()    # 同步

            start = time.time()
            # 将数据传到GPU
            obj_f, grid_f, tag_f, labels, masks, tags =\
                [data['obj_f'], data['grid_f'], data['tag_f'], data['labels'],
                 data['masks'], data['tags_calc']]
            obj_f = obj_f.cuda()
            grid_f = {i: v.cuda() for i, v in grid_f.items()}
            # tag_f = tag_f.cuda()
            tag_f = {i: v.cuda() for i, v in tag_f.items()}
            tags = tags.cuda()
            labels = labels.cuda()
            masks = masks.cuda()

            # 梯度清零
            optimizer.zero_grad()
            # 模型前向传播
            model_out = dp_lw_model(obj_f, grid_f, tag_f, labels, masks, tags,
                                    data['gts'], torch.arange(0, len(data['gts'])),
                                    SCST_flag)

            # 计算损失
            loss = model_out['loss'].mean()

            # Back prop 损失反向传播
            loss.backward()
            # Clip gradients
            if opt.grad_clip_value != 0:  # 对梯度进行裁剪，防止梯度爆炸
                # 根据'grad_clip_mode'确定裁剪模式：常见有'norm’通过限制范数；'value'限制梯度的数值范围
                # 根据'grad_clip_value‘确定裁剪阈值
                getattr(torch.nn.utils, 'clip_grad_%s_' % (opt.grad_clip_mode))(model.parameters(), opt.grad_clip_value)
            # Update parameters
            optimizer.step()
            train_loss = loss.item()
            torch.cuda.synchronize()
            end = time.time()

            # Print out log info    打印loss
            if not SCST_flag:
                print("***** iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, end - start))
            elif SCST_flag:
                print("***** iter {} (epoch {}), loss = {:.3f}, avg_reward = {:.3f}, time/batch = {:.3f}" \
                    .format(iteration, epoch, train_loss, model_out['reward'].mean(), end - start)) # 【考虑要不要print(loss)】

            # Update the iteration and epoch
            iteration += 1
            if data['bounds']['wrapped']:
                epoch += 1
                epoch_done = True

            # Write the training loss summary   写入训练损失摘要
            if (iteration % opt.losses_log_every == 0):
                tb_summary_writer.add_scalar('iter/train_loss', train_loss, iteration)
                if opt.noamopt:
                    opt.current_lr = optimizer.rate()
                elif opt.reduce_on_plateau:
                    opt.current_lr = optimizer.current_lr
                tb_summary_writer.add_scalar('iter/learning_rate', opt.current_lr, iteration)
                tb_summary_writer.add_scalar('iter/scheduled_sampling_prob', model.ss_prob, iteration)

                if SCST_flag:
                    tb_summary_writer.add_scalar('iter/avg_reward', model_out['reward'].mean(), iteration)

                histories['loss_history'][iteration] = train_loss if not SCST_flag else model_out['reward'].mean()
                histories['lr_history'][iteration] = opt.current_lr
                histories['ss_prob_history'][iteration] = model.ss_prob

            # Update the iteration and epoch
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['loader_state_dict'] = loader.state_dict()

            # save in tensorboard each epoch    # 每个epoch保存一次tensorboard，保存test的结果
            if epoch_done:
                tb_summary_writer.add_scalar('epoch/train_loss', train_loss, epoch)
                # eval model
                eval_kwargs = {'split': 'test',
                            'dataset': opt.input_json,
                            'topk': opt.topk,
                            'language_eval': 1} # data/dataset_coco.json
                eval_kwargs.update(vars(opt))
                test_loss, predictions, n_predictions, lang_stats = eval_utils.eval_split(
                    dp_model, lw_model.crit, loader, eval_kwargs)
                tb_summary_writer.add_scalar('epoch/test_loss', test_loss, epoch)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        tb_summary_writer.add_scalar(f'epoch_lang/{k}', v, epoch)
                histories['val_result_history'][epoch] = {'loss': test_loss, 'lang_stats': lang_stats, 'predictions': predictions, 'n_predictions': n_predictions}
                test_cider = lang_stats['CIDEr']
                if test_cider > best_cider:
                    best_cider = test_cider
                    infos['best_test_score'] = best_cider
                    print('Reached new val best cider %f' % best_cider, 'saving best checkpoint')
                    best_path = os.path.join('./outputs', opt.checkpoint_path, 'best_test_scores.json')
                    with open(best_path, "w") as f:
                        json.dump(lang_stats, f)

            # make evaluation on validation set, and save model  在验证集上进行评估，并保存模型
            if (iteration % opt.save_checkpoint_every == 0 and not opt.save_every_epoch) or \
                    (epoch_done and opt.save_every_epoch):
                # eval model
                eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json,
                            'topk': opt.topk} # data/dataset_coco.json
                eval_kwargs.update(vars(opt))
                val_loss, predictions, n_predictions, lang_stats = eval_utils.eval_split(
                    dp_model, lw_model.crit, loader, eval_kwargs)

                # 优化器调度
                if opt.reduce_on_plateau:
                    if 'CIDEr' in lang_stats:
                        optimizer.scheduler.step(-lang_stats['CIDEr'])
                    else:
                        optimizer.scheduler.step(val_loss)
                # Write validation result into summary  将验证结果写入摘要
                tb_summary_writer.add_scalar('iter/validation loss', val_loss, iteration)
                if lang_stats is not None:
                    for k,v in lang_stats.items():
                        tb_summary_writer.add_scalar(f'iter_lang/{k}', v, iteration)
                histories['val_result_history'][iteration] = {'loss': val_loss, 'lang_stats': lang_stats,
                                                                'predictions': predictions}

                # Save model if is improving on validation result
                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = - val_loss

                best_flag = False
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True

                # Dump miscalleous informations
                infos['best_val_score'] = best_val_score
                utils.save_checkpoint(opt, model, infos, optimizer, histories)
                if opt.save_history_ckpt:
                    utils.save_checkpoint(opt, model, infos, optimizer,
                        append=str(epoch) if opt.save_every_epoch else str(iteration))

                if best_flag:
                    utils.save_checkpoint(opt, model, infos, optimizer, append='best')

        # gc.collect()
        torch.cuda.empty_cache()    # 释放显存

    # 程序异常退出时，保存模型
    except (RuntimeError, KeyboardInterrupt):
        print('Save ckpt on exception ...')
        utils.save_checkpoint(opt, model, infos, optimizer)
        print('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)
        os._exit(0)


opt = opts.parse_opt()  # 解析参数
train(opt)  # 训练
