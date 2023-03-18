import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_

from omegaconf import OmegaConf

import utils
import backbone
import losses
from dataset.dataset_arcface import MXFaceDataset, DataLoaderX
from tricks.partial_fc import PartialFC
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_amp import MaxClipGradScaler
from utils.utils_load_from_cfg import instantiate_from_config

from dataset.dataset_aug import FaceRandOccMask, Msk2Tenser
import torchvision.transforms as transforms
from torch.autograd import Variable
from thop import profile


def main(args):

    import random
    import numpy as np
    random.seed(4)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    import mxnet as mx
    mx.random.seed(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    """ DDP Training """
    try:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        dist.init_process_group("nccl")
    except KeyError:
        world_size = 1
        rank = 0
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:12584",
            rank=rank,
            world_size=world_size,
        )
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    """ Load from cfg """
    config = OmegaConf.load(args.config)
    if args.resume >= 1:
        config = OmegaConf.load(os.path.join(args.resume_folder, os.path.split(args.config)[-1]))
    cfg_train = config.train
    output_folder = os.path.join(cfg_train.out_folder, "%s_%s" % (cfg_train.out_name, cfg_train.exp_id))
    if not os.path.exists(output_folder) and rank is 0:
        os.makedirs(output_folder)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, output_folder)
    if rank is 0 and args.resume == 0:
        os.system('cp %s %s' % (args.config, output_folder))
        logging.info('config file copied to %s.' % output_folder)
    trainset = FaceRandOccMask(
        root_dir=cfg_train.rec,
        local_rank=0,
        is_train=True,
        out_size=(112, 112),
        is_gray=False,
        use_norm=True)
    # trainset = MXFaceDataset(
    #     root_dir=cfg.rec,
    #     local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)
    cfg_train.nw = 0
    nw = cfg_train.nw
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg_train.batch_size,
        sampler=train_sampler, num_workers=nw, pin_memory=True, drop_last=True)

    from tricks.automatic_weighted_loss import AutomaticWeightedLoss
    awl = AutomaticWeightedLoss(2).cuda()

    backbone = instantiate_from_config(config.model).to(local_rank)

    if args.resume:
        backbone_pth = os.path.join(output_folder, "backbone.pth")
        backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
        if rank is 0:
            logging.info("backbone resume successfully from %s!" % args.resume_folder)
        # awl_pth = os.path.join(cfg.output, "awloss.pth")
        # if os.path.exists(awl_pth):
        #     awl.load_state_dict(torch.load(awl_pth, map_location=torch.device(local_rank)))

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank],
        find_unused_parameters=True)
    backbone.train()

    # for ps in awl.parameters():
    #     dist.broadcast(ps, 0)
    # awl.train()

    lr_adam = cfg_train.adam_lr_max
    lr_sgd = cfg_train.sgd_lr_max
    lr_scale = cfg_train.batch_size * world_size / cfg_train.base_bs
    if cfg_train.adam_lr_scale:
        lr_adam *= lr_scale
    if cfg_train.sgd_lr_scale:
        lr_sgd *= lr_scale

    opt_adam = torch.optim.AdamW(
        params=[{'params': backbone.parameters()},
                # {'params': awl.parameters(), 'weight_decay': 0}
                ],
        lr=lr_adam,
        betas=(0.9, 0.999),
        eps=1e-05,
        weight_decay=0.1,
        amsgrad=False
    )
    opt_sgd = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=lr_sgd,
        momentum=0.9, weight_decay=5e-4)

    scheduler_adam = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=opt_adam, T_max=cfg_train.adam_epoch, eta_min=cfg_train.adam_lr_min,
    )
    scheduler_sgd = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=opt_sgd, T_max=cfg_train.sgd_epoch, eta_min=cfg_train.sgd_lr_min,
    )

    full_epoch = cfg_train.adam_epoch + cfg_train.sgd_epoch
    total_step = int(len(trainset) / cfg_train.batch_size / world_size *
                     (full_epoch - args.resume))
    if rank == 0:
        logging.info("Total Step is: %d" % total_step)

    callback_verification = CallBackVerification(8000, rank, cfg_train.val_targets, cfg_train.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg_train.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, output_folder)

    loss = AverageMeter()
    loss_1 = AverageMeter()
    global_step = 0
    grad_scaler = MaxClipGradScaler(init_scale=cfg_train.batch_size,  # cfg.batch_size
                                    max_scale=128 * cfg_train.batch_size,
                                    growth_interval=100) if cfg_train.fp16 else None

    from tricks.consensus_loss import StructureConsensuLossFunction
    seg_criterion = StructureConsensuLossFunction(10.0, 5.0, 'idx', 'idx')
    cls_criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(0, full_epoch):
        train_sampler.set_epoch(epoch)
        if epoch < cfg_train.adam_epoch:
            scheduler = scheduler_adam
            opt = opt_adam
            inner_epoch = epoch
        else:
            scheduler = scheduler_sgd
            opt = opt_sgd
            inner_epoch = epoch - cfg_train.adam_epoch

        if epoch < args.resume:
            if rank is 0:
                print('=====> skip epoch %d' % (epoch))
            scheduler.step()
            continue

        for step, batch in enumerate(train_loader):
            global_step += 1

            """ (img, label), MXFaceDataset
                (img, msk, label), FaceByRandOccMask (no KD)
                (img, msk, ori, label), FaceByRandOccMask (with KD)
            """
            img, label = batch[0], batch[-1]
            msk = batch[1] if len(batch) >= 3 else None
            # ori = batch[2] if len(batch) == 4 and conf.peer_params.use_ori else None

            """ op1: full classes """
            with torch.cuda.amp.autocast(cfg_train.fp16):
                if args.network in ('dpt',):
                    final_id, msk_final = backbone(img, label)
                    with torch.no_grad():
                        msk_cc_var = Variable(msk.clone().cuda(non_blocking=True))
                    seg_loss = seg_criterion(msk_final, msk_cc_var, msk)
                elif args.network in ('ft', 'fst', 'dpt_sub'):
                    final_id = backbone(img, label)
                    seg_loss = 0.

                cls_loss = cls_criterion(final_id, label)

                l1 = 1
                total_loss = cls_loss + l1 * seg_loss
                # total_loss = awl(cls_loss, seg_loss, rank=rank)

            if cfg_train.fp16:
                grad_scaler.scale(total_loss).backward()
                grad_scaler.unscale_(opt)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                grad_scaler.step(opt)
                grad_scaler.update()
            else:
                total_loss.backward()
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                opt.step()
            opt.zero_grad()

            loss_v = total_loss
            """ end - CE loss """

            loss_1.update(cls_loss, 1)
            if global_step % 100 == 0 and rank == 0:
                logging.info('[%s], seg_loss=%.4f, cls_loss=%.4f, lr_adam=%.4f(scale:%s), lr_sgd=%.4f(scale:%s),'
                      'num_workers=%d, bs=%d*%d'
                      % (output_folder, seg_loss, loss_1.avg,
                         cfg_train.adam_lr_max, cfg_train.adam_lr_scale,
                         cfg_train.sgd_lr_max, cfg_train.sgd_lr_scale,
                         nw, cfg_train.batch_size, world_size))
                loss_1.reset()

            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, cfg_train.fp16, grad_scaler)
            callback_verification(global_step, backbone)

            if global_step % 1000 == 0:
                for param_group in opt.param_groups:
                    lr = param_group['lr']
                print(lr)

        callback_checkpoint(global_step, backbone, is_adam_last=epoch == cfg_train.adam_epoch - 1,
                            partial_fc=None, awloss=None,)
        scheduler.step()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DualPathTransformer Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network', type=str, default='ft', help='network type')
    parser.add_argument('--config', type=str, default='configs/train_ft.yaml', help='config path')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    parser.add_argument('--resume_folder', type=str, default='', help='model resuming folder')
    args_ = parser.parse_args()
    main(args_)
