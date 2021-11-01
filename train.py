import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_

import utils
import backbone
import losses
from config import cfg
from dataset.dataset_arcface import MXFaceDataset, DataLoaderX
from tricks.partial_fc import PartialFC
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_amp import MaxClipGradScaler

from dataset.dataset_aug import FaceRandOccMask, Msk2Tenser
import torchvision.transforms as transforms
from torch.autograd import Variable
from thop import profile

torch.backends.cudnn.benchmark = True


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
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    dist_url = "tcp://{}:{}".format(os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"])
    dist.init_process_group(backend='nccl', init_method=dist_url, rank=rank, world_size=world_size)
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)

    if not os.path.exists(cfg.output) and rank is 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)
    img_tf_train = transforms.Compose(
        [transforms.ToTensor(), ])
    mask_tf = transforms.Compose([Msk2Tenser(), ])
    trainset = FaceRandOccMask(
        root_dir=cfg.rec,
        local_rank=0,
        img_transform=img_tf_train,
        msk_transform=mask_tf,
        is_train=True,
        out_size=112,
        gray=False,
        norm=True)
    # trainset = MXFaceDataset(
    #     root_dir=cfg.rec,
    #     local_rank=local_rank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)
    nw = 8
    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=nw, pin_memory=True, drop_last=True)

    dropout = 0.4 if cfg.dataset is "webface" else 0
    backbone = eval("backbone.{}".format(args.network))(False,
                                                        fp16=cfg.fp16,
                                                        num_classes=cfg.num_classes,
                                                        dim=cfg.model_set.dim,
                                                        depth=cfg.model_set.depth,
                                                        heads=cfg.model_set.heads,
                                                        mlp_dim=cfg.model_set.mlp_dim,
                                                        emb_dropout=cfg.model_set.emb_dropout,
                                                        dim_head=cfg.model_set.dim_head,
                                                        dropout=cfg.model_set.dropout
                                                        ).to(local_rank)

    if args.resume:
        try:
            backbone_pth = os.path.join(cfg.output, "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank is 0:
                logging.info("backbone resume successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("resume fail, backbone init successfully!")

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)
    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank],
        find_unused_parameters=True)
    backbone.train()

    margin_softmax = eval("losses.{}".format(args.loss))()
    module_partial_fc = PartialFC(
        rank=rank, local_rank=local_rank, world_size=world_size, resume=args.resume,
        batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output)

    # opt_backbone = torch.optim.SGD(
    #     params=[{'params': backbone.parameters()}],
    #     lr=cfg.lr / 512 * cfg.batch_size * world_size,
    #     momentum=0.9, weight_decay=cfg.weight_decay)
    # opt_backbone = torch.optim.Adam(
    #     params=[{'params': backbone.parameters()}],
    #     lr=cfg.lr / 512 * cfg.batch_size * world_size,
    #     betas=(0.9, 0.999),
    #     eps=1e-08,
    #     weight_decay=0.04,
    #     amsgrad=False
    # )
    opt_backbone = torch.optim.AdamW(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.04,
        amsgrad=False
    )
    opt_pfc = torch.optim.SGD(
        params=[{'params': module_partial_fc.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    scheduler_pfc = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_pfc, lr_lambda=cfg.lr_func)

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank is 0: logging.info("Total Step is: %d" % total_step)

    callback_verification = CallBackVerification(8000, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss = AverageMeter()
    global_step = 0
    grad_scaler = MaxClipGradScaler(init_scale=cfg.batch_size,  # cfg.batch_size
                                    max_scale=128 * cfg.batch_size,
                                    growth_interval=100) if cfg.fp16 else None

    from tricks.consensus_loss import StructureConsensuLossFunction
    seg_criterion = StructureConsensuLossFunction(10.0, 5.0, 'idx', 'idx')
    cls_criterion = torch.nn.CrossEntropyLoss()

    from torch.cuda import amp
    scaler = amp.GradScaler(init_scale=cfg.batch_size,
                            growth_interval=100, enabled=cfg.fp16)

    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for step, (img, msk, label) in enumerate(train_loader):
        # for step, (img, label) in enumerate(train_loader):
            global_step += 1
            # if global_step % 100 == 0:
            #     print('rank:', rank, time.strftime("[%Y-%m-%d-%H_%M_%S]", time.localtime()), global_step)

            """ op1: full classes """
            with amp.autocast(cfg.fp16):
                # [f_id, msk_final] = backbone(img)
                final_id = backbone(img, label)
                # f_id = F.normalize(f_id)  # TODO: close normalize

                """ 1. occ """
                # with torch.no_grad():
                #     msk_cc_var = Variable(msk.clone().cuda(non_blocking=True))
                # seg_loss = seg_criterion(msk_final, msk_cc_var, msk)
                seg_loss = 0.

                """ 2. id """
                cls_loss = cls_criterion(final_id, label)

                l1 = 3
                total_loss = cls_loss + l1 * seg_loss

            if cfg.fp16:
                grad_scaler.scale(total_loss).backward()
                grad_scaler.unscale_(opt_backbone)
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                grad_scaler.step(opt_backbone)
                grad_scaler.update()
            else:
                total_loss.backward()
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                opt_backbone.step()
            opt_backbone.zero_grad()

            loss_v = total_loss
            """ end - CE loss """

            """ op2. partial fc """
            # features = backbone(img)
            # # features = F.normalize(backbone(img))  # CosFace needs normalize
            # # from torchinfo import summary
            # # summary(backbone, input_size=(1, 3, 112, 112))
            # x_grad, loss_v = module_partial_fc.forward_backward(label, features, opt_pfc)
            # if cfg.fp16:
            #     features.backward(grad_scaler.scale(x_grad))
            #     grad_scaler.unscale_(opt_backbone)
            #     clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
            #     grad_scaler.step(opt_backbone)
            #     grad_scaler.update()
            # else:
            #     features.backward(x_grad)
            #     clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
            #     opt_backbone.step()
            #
            # opt_pfc.step()
            # module_partial_fc.update()
            #
            # # if global_step % 50 == 3:
            # #     for name, params in backbone.named_parameters():
            # #         if params.grad is not None:
            # #             logging.info('-->name: %s, -->grad_val: %f,'
            # #                          '-->max: %f, -->min:%f'
            # #                          '' % (name, params.grad.data.cpu().mean(),
            # #                                params.data.cpu().max(),
            # #                                params.data.cpu().min()))
            # #         else:
            # #             logging.info('-->name: %s, -->grad: None' % (name))
            #
            # opt_backbone.zero_grad()
            # opt_pfc.zero_grad()
            # seg_loss = 0.
            # cls_loss = loss_v
            # l1 = 0.
            """ end - partial fc"""

            if global_step % 100 == 0 and rank == 0:
                print('[exp_%d], seg_loss=%.4f, cls_loss=%.4f, scale=%.4f, lr=%.4f, l1=%.4f, '
                      'num_workers=%d'
                      % (cfg.exp_id, seg_loss, cls_loss, scaler.get_scale(), cfg.lr, l1, nw))

            loss.update(loss_v, 1)
            callback_logging(global_step, loss, epoch, cfg.fp16, grad_scaler)
            callback_verification(global_step, backbone)

            if global_step % 1000 == 0:
                for param_group in opt_backbone.param_groups:
                    lr = param_group['lr']
                print(lr)

        callback_checkpoint(global_step, backbone, module_partial_fc)
        scheduler_backbone.step()
        # scheduler_pfc.step()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DualPathTransformer Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--network', type=str, default='dpt_r18s3_ca3', help='backbone network')
    parser.add_argument('--loss', type=str, default='ArcFace', help='loss function')
    parser.add_argument('--resume', type=int, default=0, help='model resuming')
    args_ = parser.parse_args()
    main(args_)
