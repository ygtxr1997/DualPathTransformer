from easydict import EasyDict as edict

cfg = edict()
cfg.dataset = "ms1m-retinaface-t2"
cfg.embedding_size = 512
cfg.sample_rate = 1
cfg.fp16 = True
cfg.momentum = 0.9
cfg.weight_decay = 5e-4
cfg.batch_size = 128  # 128
cfg.lr = 3e-4  # 0.1 for batch size is 512

cfg.nw = 20
cfg.l1 = 1

"""
0: Softmax; 1: ArcFace; 2: CosFace; 3: AMArcFace; 4: AMCosFace
"""
cfg.loss_type = 4
cfg.am_a = 1.2
cfg.am_k = 0.1

""" Setting EXP ID """
cfg.exp_id = 0
cfg.output = "out/tmp_" + str(cfg.exp_id)
print('output path: ', cfg.output)

""" Setting for Model FaceTransformer """
ft_set = edict()
ft_set.dim = 512
ft_set.depth = 4
ft_set.heads = 8
ft_set.dim_head = 64
ft_set.mlp_dim = 512
ft_set.emb_dropout = 0.1
ft_set.dropout = 0.1

""" Setting for Model SegTransformer """
st_set = edict()
st_set.dim = 512
st_set.depth = 1
st_set.heads = 4
st_set.dim_head = 32
st_set.mlp_dim = 128
st_set.emb_dropout = 0.2
st_set.dropout = 0.2

""" Setting for Model DualPathTransformer"""
dp_set = edict()
dp_set.dim = 384
dp_set.depth = 2
dp_set.heads_id = 6
dp_set.heads_oc = 6
dp_set.dim_head_id = 64
dp_set.dim_head_oc = 64
dp_set.mlp_dim_id = 192
dp_set.mlp_dim_oc = 192
dp_set.emb_dropout = 0.1
dp_set.dropout_id = 0.1
dp_set.dropout_oc = 0.1

""" Setting for Model DPT Only SA """
dptsa_set = edict()
dptsa_set.dim = 256
dptsa_set.depth = 1
dptsa_set.heads = 12
dptsa_set.dim_head = 64
dptsa_set.mlp_dim = 384
dptsa_set.emb_dropout = 0.1
dptsa_set.dropout = 0.1

cfg.model_set = dp_set
print(cfg.model_set)

if cfg.dataset == "emore":
    cfg.rec = "/train_tmp/faces_emore"
    cfg.num_classes = 85742
    cfg.num_image = 5822653
    cfg.num_epoch = 16
    cfg.warmup_epoch = -1
    cfg.val_targets = ["lfw", ]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14] if m - 1 <= epoch])
    cfg.lr_func = lr_step_func

elif cfg.dataset == "ms1m-retinaface-t2":
    cfg.rec = "/home/yuange/dataset/ms1m-retinaface"
    import os
    if os.path.exists("/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/datasets/ms1m-retinaface"):
        cfg.rec = "/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/datasets/ms1m-retinaface"
        cfg.nw = 14
    elif os.path.exists("/tmp/train_tmp"):
        cfg.rec = "/tmp/train_tmp/ms1m-retinaface"  # mount on RAM
        cfg.nw = 12
    cfg.num_classes = 93431  # 91180
    cfg.num_epoch = 25
    cfg.warmup_epoch = -10  # -1
    cfg.val_targets = ["lfw", "cfp_fp", ]  # ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < cfg.warmup_epoch else 0.1 ** len(
            [m for m in [11, 17, 22] if m - 1 <= epoch])  # 0.1, 0.01, 0.001, 0.0001

    import numpy as np
    cfg.min_lr = 1e-4
    def lr_fun_cos(cur_epoch):
        """Cosine schedule (cfg.OPTIM.LR_POLICY = 'cos')."""
        lr = 0.5 * (1.0 + np.cos(np.pi * cur_epoch / cfg.num_epoch))
        return (1.0 - cfg.min_lr) * lr + cfg.min_lr

    cfg.warmup_factor = 0.3
    def lr_step_func_cos(epoch):
        cur_lr = lr_fun_cos(cur_epoch=epoch) * cfg.lr
        if epoch < cfg.warmup_epoch:
            alpha = epoch / cfg.warmup_epoch
            warmup_factor = cfg.warmup_factor * (1.0 - alpha) + alpha
            cur_lr *= warmup_factor
        return lr_fun_cos(cur_epoch=epoch)
        # return cur_lr / cfg.lr

    cfg.lr_func = lr_step_func

elif cfg.dataset == "glint360k":
    # make training faster
    # our RAM is 256G
    # mount -t tmpfs -o size=140G  tmpfs /train_tmp
    cfg.rec = "/train_tmp/glint360k"
    cfg.num_classes = 360232
    cfg.num_image = 17091657
    cfg.num_epoch = 20
    cfg.warmup_epoch = -1
    cfg.val_targets = ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < cfg.warmup_epoch else 0.1 ** len(
            [m for m in [8, 12, 15, 18] if m - 1 <= epoch])
    cfg.lr_func = lr_step_func

elif cfg.dataset == "webface":
    cfg.rec = "/train_tmp/faces_webface_112x112"
    cfg.num_classes = 10572
    cfg.num_image = "forget"
    cfg.num_epoch = 34
    cfg.warmup_epoch = -1
    cfg.val_targets = ["lfw", "cfp_fp", "agedb_30"]

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < cfg.warmup_epoch else 0.1 ** len(
            [m for m in [20, 28, 32] if m - 1 <= epoch])
    cfg.lr_func = lr_step_func

