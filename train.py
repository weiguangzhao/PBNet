# _*_coding : UTF-8_*_
# Code writer: Weiguang.Zhao
# Writing time: 2022/6/28  下午10:34
# File Name: train.py
# IDE: PyCharm

import os, sys
import time
import random
import torch
import numpy as np
import torch.optim as optim


from math import cos, pi
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from config.config import get_parser
import tools.log as log
from tools.mIOU import intersectionAndUnionGPU, non_max_suppression


# Epoch counts from 0 to N-1
def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_epoch(train_loader, model, model_fn, optimizer, epoch):
    model.train()

    # #for log the run time and remain time
    iter_time = log.AverageMeter()
    batch_time = log.AverageMeter()
    start_time = time.time()
    end_time = time.time()  # initialization
    am_dict = {}

    # #start train
    for i, batch in enumerate(train_loader):
        torch.cuda.empty_cache()
        batch_time.update(time.time() - end_time)  # update time

        cosine_lr_after_step(optimizer, cfg.lr, epoch, cfg.step_epoch, cfg.epochs, clip=1e-6)  # adjust lr

        # #loss, result, visual_dict , meter_dict (visual_dict: tensorboardX, meter_dict: average batch loss)
        loss, _, visual_dict, meter_dict = model_fn(batch, model, epoch, cfg, task='train')

        # # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # #average batch loss, time for print
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = log.AverageMeter()
            am_dict[k].update(v[0], v[1])

        current_iter = epoch * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter
        iter_time.update(time.time() - end_time)
        end_time = time.time()
        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
            sys.stdout.write("epoch: {}/{} iter: {}/{} loss: {:.4f}({:.4f})  data_time: {:.2f}({:.2f}) "
                             "iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n"
                             .format(epoch, cfg.epochs, i + 1, len(train_loader), am_dict['loss'].val,
                                     am_dict['loss'].avg,
                                     batch_time.val, batch_time.avg, iter_time.val, iter_time.avg,
                                     remain_time=remain_time))
            if (i == len(train_loader) - 1): print()

    if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
        logger.info("epoch: {}/{}, train loss: {:.4f},  time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg,
                                                                          time.time() - start_time))
        # #write tensorboardX
        lr = optimizer.param_groups[0]['lr']
        for k in am_dict.keys():
            if k in visual_dict.keys():
                writer.add_scalar(k + '_train', am_dict[k].avg, epoch)
                writer.add_scalar('train/learning_rate', lr, epoch)

        # # save pretrained model
        pretrain_file = log.checkpoint_save(model, optimizer, cfg.logpath, epoch, cfg.save_freq)
        logger.info('Saving {}'.format(pretrain_file))
    pass


def eval_epoch(val_loader, model, model_fn, epoch):
    if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}
    gt_dir = 'datasets/scannetv2'
    semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    with torch.no_grad():
        model.eval()
        start_time = time.time()

        intersection_meter = log.AverageMeter()
        union_meter = log.AverageMeter()
        target_meter = log.AverageMeter()
        All_accm = log.AverageMeter()
        Tp_accm = log.AverageMeter()
        Tf_accm = log.AverageMeter()
        matches = {}
        for i, batch in enumerate(val_loader):
            torch.cuda.empty_cache()
            loss, pred, visual_dict, meter_dict = model_fn(batch, model, epoch, cfg, task='eval')
            # #==========================================sem eval=========================================
            pred_sem = pred['sem']
            sem_label = batch['sem'].type(torch.int64).cuda()
            intersection, union, target = intersectionAndUnionGPU(pred_sem.detach().clone(), sem_label.detach().clone(), cfg.sem_num,-100)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)

            # #average batch loss, time for print
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = log.AverageMeter()
                am_dict[k].update(v[0], v[1])
            if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
                sys.stdout.write(
                    "\riter: {}/{} loss: {:.4f}({:.4f}) Accuracy {accuracy:.4f} ".format(i + 1, len(val_loader),
                                                                                         am_dict['loss'].val,
                                                                                         am_dict['loss'].avg,
                                                                                         accuracy=accuracy))
                if (i == len(val_loader) - 1): print()
        if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
            logger.info("epoch: {}/{}, val loss: {:.4f},  time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg,
                                                                            time.time() - start_time))

            # #write tensorboardX
            for k in am_dict.keys():
                if k in visual_dict.keys():
                    writer.add_scalar(k + '_eval', am_dict[k].avg, epoch)
        # #calculate ACC
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

        if (cfg.dist and cfg.local_rank == 0) or cfg.dist == False:
            logger.info('mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
            # #write tensorboardX
            writer.add_scalar('val/mIOU_eval', mIoU, epoch)
            writer.add_scalar('val/mAcc_eval', mAcc, epoch)
            writer.add_scalar('val/allACC_eval', allAcc, epoch)


def Distributed_training(gpu, cfgs):
    global cfg
    cfg = cfgs
    cfg.local_rank = gpu
    # logger and summary write
    if cfg.local_rank == 0:
        # logger
        global logger
        from tools.log import get_logger
        logger = get_logger(cfg)
        logger.info(cfg)  # log config
        # summary writer
        global writer
        writer = SummaryWriter(cfg.logpath)
    cfg.rank = cfg.node_rank * cfg.gpu_per_node + gpu
    print('[PID {}] rank: {}  world_size: {}'.format(os.getpid(), cfg.rank, cfg.world_size))
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d' % cfg.tcp_port, world_size=cfg.world_size,
                            rank=cfg.rank)
    if cfg.local_rank == 0:
        logger.info(cfg)
    # #set cuda
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    torch.cuda.set_device(gpu)
    if cfg.local_rank == 0:
        logger.info('cuda available: {}'.format(use_cuda))

    # #create model
    if cfg.local_rank == 0:
        logger.info('=> creating model ...')
    from network.PBNet import PBNet as net
    from network.PBNet import model_fn
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    model = net(cfg)
    model = model.to(gpu)
    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    if cfg.local_rank == 0:
        logger.info('#Model parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    #  #optimizer
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr,
                              momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, betas=(0.9, 0.99),
                                weight_decay=cfg.weight_decay)
    # load dataset
    if cfg.dataset == 'Scannet':
        from datasets.scannetv2.dataset_preprocess import Dataset
    else:
        print('do not support this dataset at present')


    dataset = Dataset(cfg)
    dataset.trainLoader()
    dataset.valLoader()
    if cfg.local_rank == 0:
        logger.info('Training samples: {}'.format(len(dataset.train_file_list)))
        logger.info('Validation samples: {}'.format(len(dataset.val_file_list)))

    # #train
    cfg.pretrain = ''  # Automatically identify breakpoints
    start_epoch, pretrain_file = log.checkpoint_restore(model, None, cfg.logpath, dist=cfg.dist, pretrain_file=cfg.pretrain,
                                                        gpu=gpu)
    if cfg.local_rank == 0:
        logger.info('Restore from {}'.format(pretrain_file) if len(pretrain_file) > 0
                    else 'Start from epoch {}'.format(start_epoch))

    for epoch in range(start_epoch, cfg.epochs):
        dataset.train_sampler.set_epoch(epoch)
        train_epoch(dataset.train_data_loader, model, model_fn, optimizer, epoch)

        # # #validation
        if cfg.validation and epoch%4==0:
            dataset.val_sampler.set_epoch(epoch)
            eval_epoch(dataset.val_data_loader, model, model_fn, epoch)
    pass


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
    cfg = get_parser()
    # # fix seed for debug
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)


    # # Determine whether it is distributed training
    cfg.world_size = cfg.nodes * cfg.gpu_per_node
    cfg.dist = True if cfg.world_size > 1 else False
    if cfg.dist:
        mp.spawn(Distributed_training, nprocs=cfg.gpu_per_node, args=(cfg,))
    else:
        Single_card_training(cfg.local_rank, cfg)