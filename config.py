from easydict import EasyDict as edict

config = edict()
config.dataset = "ms1m-retinaface-t2"
config.embedding_size = 512
config.sample_rate = 1
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128  # 128
config.lr = 1e-4  # 0.1 for batch size is 512

config.exp_id = 20
config.output = "tmp_" + str(config.exp_id)
print('output path: ', config.output)


if config.dataset == "emore":
    config.rec = "/train_tmp/faces_emore"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch = 16
    config.warmup_epoch = -1
    config.val_targets = ["lfw", ]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "ms1m-retinaface-t2":
    config.rec = "/home/yuange/dataset/ms1m-retinaface"
    import os
    if os.path.exists("/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/datasets/ms1m-retinaface"):
        config.rec = "/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/datasets/ms1m-retinaface"
    config.num_classes = 93431 # 91180
    config.num_epoch = 25
    config.warmup_epoch = -10 # -1
    config.val_targets = ["lfw", "cfp_fp", ]  # ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [11, 17, 22] if m - 1 <= epoch])  # 0.1, 0.01, 0.001, 0.0001

    import numpy as np
    config.min_lr = 0
    def lr_fun_cos(cur_epoch):
        """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
        lr = 0.5 * (1.0 + np.cos(np.pi * cur_epoch / config.num_epoch))
        return (1.0 - config.min_lr) * lr + config.min_lr

    config.warmup_factor = 0.3
    def lr_step_func_cos(epoch):
        cur_lr = lr_fun_cos(cur_epoch=epoch) * config.lr
        if epoch < config.warmup_epoch:
            alpha = epoch / config.warmup_epoch
            warmup_factor = config.warmup_factor * (1.0 - alpha) + alpha
            cur_lr *= warmup_factor
        return lr_fun_cos(cur_epoch=epoch)
        # return cur_lr / config.lr

    config.lr_func = lr_step_func

elif config.dataset == "glint360k":
    # make training faster
    # our RAM is 256G
    # mount -t tmpfs -o size=140G  tmpfs /train_tmp
    config.rec = "/train_tmp/glint360k"
    config.num_classes = 360232
    config.num_image = 17091657
    config.num_epoch = 20
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [8, 12, 15, 18] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "webface":
    config.rec = "/train_tmp/faces_webface_112x112"
    config.num_classes = 10572
    config.num_image = "forget"
    config.num_epoch = 34
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [20, 28, 32] if m - 1 <= epoch])
    config.lr_func = lr_step_func
