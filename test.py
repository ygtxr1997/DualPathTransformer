# import torch
#
# import config as cfg
# from thop import profile
#
# img = torch.randn(1, 3, 112, 112)
#
# """ 1. Dual-Path Transformer """
# from backbone.dual_path_transformer import DualPathTransformer
# dpt = DualPathTransformer(cnn_layers=[3, 4, 3],
#                           dim=cfg.dp_set.dim,
#                           depth=cfg.dp_set.depth,
#                           heads_id=cfg.dp_set.heads_id,
#                           heads_oc=cfg.dp_set.heads_oc,
#                           mlp_dim_id=cfg.dp_set.mlp_dim_id,
#                           mlp_dim_oc=cfg.dp_set.mlp_dim_oc,
#                           num_classes=93431,
#                           dim_head_id=cfg.dp_set.dim_head_id,
#                           dim_head_oc=cfg.dp_set.dim_head_oc,
#                           dropout_oc=0.2,
#                           fp16=False).to(device)
# # feature_id, feature_oc = dpt(img.to(device))
# # print(feature_id.shape, feature_oc.shape)
#
# """ 2. Face Transformer """
# from backbone.face_transformer import FaceTransformer
# ft = FaceTransformer(cnn_layers=[2, 2, 2],
#                      num_classes=93431,
#                      dim=cfg.ft_set.dim,
#                      depth=cfg.ft_set.depth,
#                      heads=cfg.ft_set.heads,
#                      mlp_dim=cfg.ft_set.mlp_dim,
#                      emb_dropout=0.,
#                      dim_head=cfg.ft_set.dim_head,
#                      dropout=0.,
#                      fp16=False).to(device)
# # feature_id = ft(img.to(device))
# # print(feature_id.shape)
#
# """ 3. IResnet + MLP """
# from backbone.iresnet import iresnet18, iresnet34, iresnet50, iresnet100
# from torch import nn
# class iresnet_mlp(nn.Module):
#     def __init__(self):
#         super(iresnet_mlp, self).__init__()
#         self.iresnet = iresnet34(False)
#         self.mlp = nn.Linear(512, 93431)
#     def forward(self, x):
#         x = self.iresnet(x)
#         x = self.mlp(x)
#         return x
#
# irnet = iresnet_mlp().to(device)
#
# """ 4. Segment Transformer """
# from backbone.seg_transformer import SegTransformer
# st = SegTransformer(cnn_layers=[2, 2, 2],
#                      num_classes=93431,
#                      dim=512,
#                      depth=1,
#                      heads=4,
#                      mlp_dim=256,
#                      emb_dropout=0.,
#                      dim_head=64,
#                      dropout=0.,
#                      fp16=False).to(device)
#
# """ 5. DPT only SA """
# from backbone.dual_path_transformer_only_sa import DualPathTransformerSA
# dptsa = DualPathTransformerSA(cnn_layers=[3, 4, 3],
#                     num_classes=93431,
#                     dim=cfg.dptsa_set.dim,
#                     depth=cfg.dptsa_set.depth,
#                     heads=cfg.dptsa_set.heads,
#                     mlp_dim=cfg.dptsa_set.mlp_dim,
#                     emb_dropout=0.,
#                     dim_head=cfg.dptsa_set.dim_head,
#                     dropout=0.,
#                     fp16=False).to(device)
#
# """ =================== flops&params ====================== """
# import backbone
#
# model = ft
# macs, params = profile(model,
#                        inputs=(img.to(device), ),
#                        custom_ops={
#                            backbone.face_transformer.Attention: backbone.face_transformer.Attention.cnt_flops,
#                            backbone.face_transformer.FeedForward: backbone.face_transformer.FeedForward.cnt_flops,
#                            backbone.seg_transformer.Attention: backbone.seg_transformer.Attention.cnt_flops,
#                            backbone.seg_transformer.FeedForward: backbone.seg_transformer.FeedForward.cnt_flops,
#                            backbone.dual_path_transformer.Attention: backbone.dual_path_transformer.Attention.cnt_flops,
#                            backbone.dual_path_transformer.FeedForward: backbone.dual_path_transformer.FeedForward.cnt_flops,
#                            backbone.dual_path_transformer.CrossAttention: backbone.dual_path_transformer.CrossAttention.cnt_flops,
#                            backbone.dual_path_transformer_only_sa.Attention: \
#                                 backbone.dual_path_transformer_only_sa.Attention.cnt_flops,
#                            backbone.dual_path_transformer_only_sa.FeedForward: \
#                                 backbone.dual_path_transformer_only_sa.FeedForward.cnt_flops,
#                        },)
# from thop import clever_format
# macs, params = clever_format([macs, params], "%.2f")
# print('MACs:', macs, 'wrong:', params)
#
# params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('%.2fM' % (params / 1e6))
#
# import time
# time.sleep(5)
import os.path
import time
import torch
import thop
from tqdm import tqdm
from omegaconf import OmegaConf

import backbone
from utils.utils_load_from_cfg import instantiate_from_config


def fet_test():
    config = OmegaConf.load('configs/train_fet12g.yaml')
    print('config is:', config)
    model = instantiate_from_config(config.model)
    model.eval()
    model = model.backbone
    print('Model loaded.')

    with torch.no_grad():
        from fvcore.nn import FlopCountAnalysis, parameter_count, parameter_count_table
        # n = 197
        # d = 256
        # feat = torch.randn(1, n, d)
        #
        # print("*" * 20, 'VanillaAttention', "*" * 20)
        # module = backbone.face_transformer.Attention(d)
        # print('out:', module(feat).shape)
        # flops = FlopCountAnalysis(module, feat)
        # print("[fvcore] FLOPs: %.2fM" % (flops.total() / 1e6))
        # print("[fvcore] #Params: %.2fM" % (parameter_count(module)[''] / 1e6))
        #
        # # macs, params = thop.profile(module, inputs=(feat,),
        # #                             custom_ops={
        # #                                 backbone.face_transformer.Attention: backbone.face_transformer.Attention.cnt_flops,
        # #                                 backbone.face_transformer.FeedForward: backbone.face_transformer.FeedForward.cnt_flops,
        # #                                 backbone.seg_transformer.Attention: backbone.seg_transformer.Attention.cnt_flops,
        # #                                 backbone.seg_transformer.FeedForward: backbone.seg_transformer.FeedForward.cnt_flops,
        # #                                 backbone.dual_path_transformer.Attention: backbone.dual_path_transformer.Attention.cnt_flops,
        # #                                 backbone.dual_path_transformer.FeedForward: backbone.dual_path_transformer.FeedForward.cnt_flops,
        # #                                 backbone.dual_path_transformer.CrossAttention: backbone.dual_path_transformer.CrossAttention.cnt_flops,
        # #                                 backbone.dual_path_transformer_only_sa.Attention: \
        # #                                     backbone.dual_path_transformer_only_sa.Attention.cnt_flops,
        # #                                 backbone.dual_path_transformer_only_sa.FeedForward: \
        # #                                     backbone.dual_path_transformer_only_sa.FeedForward.cnt_flops,
        # #                             }, verbose=False)
        # # from thop import clever_format
        # # macs, params = clever_format([macs, params], "%.2f")
        # # print('[thop] GFLOPs:', macs, 'wrong:', params)
        # params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        # print('[numel] #Params: %.2fM' % (params / 1e6))
        #
        # print("*" * 20, 'Stem Stage', "*" * 20)
        # img = torch.randn(1, 3, 112, 112)
        # module = backbone.face_transformer.EarlyConv(48, 4, up_sample=2)
        # print('out:', module(img).shape)
        # flops = FlopCountAnalysis(module, img)
        # print("[fvcore] FLOPs: %.2fM" % (flops.total() / 1e6))
        # print("[fvcore] #Params: %.2fM" % (parameter_count(module)[''] / 1e6))

        print("*" * 20, 'Full Model', "*" * 20)
        img = torch.randn(1, 3, 112, 112)
        flops = FlopCountAnalysis(model, img)
        # print("[fvcore] #Params Table\n", parameter_count_table(model, max_depth=7))
        print("[fvcore] FLOPs: %.2fG" % (flops.total() / 1e9))
        print("[fvcore] #Params: %.2fM" % (parameter_count(model)[''] / 1e6))
        # macs, params = thop.profile(model, inputs=(img, ),
        #                             custom_ops={
        #                                backbone.face_transformer.Attention: backbone.face_transformer.Attention.cnt_flops,
        #                                backbone.face_transformer.FeedForward: backbone.face_transformer.FeedForward.cnt_flops,
        #                                backbone.seg_transformer.Attention: backbone.seg_transformer.Attention.cnt_flops,
        #                                backbone.seg_transformer.FeedForward: backbone.seg_transformer.FeedForward.cnt_flops,
        #                                backbone.dual_path_transformer.Attention: backbone.dual_path_transformer.Attention.cnt_flops,
        #                                backbone.dual_path_transformer.FeedForward: backbone.dual_path_transformer.FeedForward.cnt_flops,
        #                                backbone.dual_path_transformer.CrossAttention: backbone.dual_path_transformer.CrossAttention.cnt_flops,
        #                                backbone.dual_path_transformer_only_sa.Attention: \
        #                                     backbone.dual_path_transformer_only_sa.Attention.cnt_flops,
        #                                backbone.dual_path_transformer_only_sa.FeedForward: \
        #                                     backbone.dual_path_transformer_only_sa.FeedForward.cnt_flops,
        #                             },)
        # from thop import clever_format
        # macs, params = clever_format([macs, params], "%.2f")
        # print('GFLOPs:', macs, 'wrong:', params)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[numel] #Params: %.2fM' % (params / 1e6))

    use_label = True
    bs = 8
    img = torch.randn(bs, 3, 112, 112).to(device)
    label = torch.zeros((bs), dtype=torch.long).to(device)
    model = model.to(device)
    # start = time.time()
    # for _ in tqdm(range(200)):
    #     out = model(img, label) if use_label else model(img)
    # print('Throughput: %.2f samples/sec' % (bs * 200 / (time.time() - start)))

    model.train()
    output = model(img, label) if use_label else model(img)
    output.mean().backward()
    print('Out shape:', output.shape)


def fvit_test():
    config = OmegaConf.load('configs/train_fvit04g.yaml')
    print('config is:', config)
    model = instantiate_from_config(config.model)
    model.eval()
    model = model.backbone
    # print(config.train.get('save_each', False))
    print('Model loaded.')

    with torch.no_grad():
        from fvcore.nn import FlopCountAnalysis, parameter_count, parameter_count_table
        print("*" * 20, 'Full Model', "*" * 20)
        img = torch.randn(1, 3, 112, 112)
        flops = FlopCountAnalysis(model, img)
        # print("[fvcore] #Params Table\n", parameter_count_table(model, max_depth=7))
        print("[fvcore] FLOPs: %.2fG" % (flops.total() / 1e9))
        print("[fvcore] #Params: %.2fM" % (parameter_count(model)[''] / 1e6))
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[numel] #Params: %.2fM' % (params / 1e6))

    use_label = False
    bs = 8
    img = torch.randn(bs, 3, 112, 112).to(device)
    label = torch.zeros((bs), dtype=torch.long).to(device)
    model = model.to(device)
    # start = time.time()
    # for _ in tqdm(range(200)):
    #     out = model(img, label) if use_label else model(img)
    # print('Throughput: %.2f samples/sec' % (bs * 200 / (time.time() - start)))

    model.train()
    output = model(img, label) if use_label else model(img)
    output.mean().backward()
    print('Out shape:', output.shape)


def ddp_test():
    import os
    import argparse
    import torch.distributed as dist

    """ python3 -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    --master_addr="127.0.0.1" --master_port=1234 test.py
    """

    parser = argparse.ArgumentParser(description='PyTorch DualPathTransformer Training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    args = parser.parse_args()

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

    config = OmegaConf.load('configs/train_ft.yaml')
    print('config is:', config)
    model = instantiate_from_config(config.model)
    model.train()
    model.to(device)
    print('Model loaded.')

    use_label = True
    bs = 2
    img = torch.randn(bs, 3, 112, 112).to(device)
    label = torch.zeros((bs), dtype=torch.long).to(device)

    output = model(img, label) if use_label else model(img)
    output.mean().backward()
    print('Out shape:', output.shape)
    exit()


def opn_test():
    config = OmegaConf.load('configs/opn50.yaml')
    print('config is:', config)
    model: torch.nn.Module = instantiate_from_config(config.model)
    model.eval()
    # out_folder = os.path.join(config.train.out_folder,
    #                           "%s_%d" % (config.train.out_name, config.train.exp_id))
    # weight = torch.load(os.path.join(out_folder, 'backbone.pth'))
    # model.load_state_dict(weight)
    # print('OPN Model loaded.')

    # print(model.transformer.layers[0][0].fn.to_qkv.weight)

    with torch.no_grad():
        from fvcore.nn import FlopCountAnalysis, parameter_count, parameter_count_table

        print("*" * 20, 'Full Model', "*" * 20)
        img = torch.randn(1, 3, 112, 112)
        flops = FlopCountAnalysis(model, img)
        # print("[fvcore] #Params Table\n", parameter_count_table(model, max_depth=7))
        print("[fvcore] FLOPs: %.2fG" % (flops.total() / 1e9))
        print("[fvcore] #Params: %.2fM" % (parameter_count(model)[''] / 1e6))

    use_label = False
    bs = 8
    img = torch.randn(bs, 3, 112, 112).to(device)
    label = torch.zeros((bs), dtype=torch.long).to(device)
    model = model.to(device)
    start = time.time()
    for _ in tqdm(range(200)):
        out = model(img, label) if use_label else model(img)
    print('Throughput: %.2f samples/sec' % (bs * 200 / (time.time() - start)))


def dpt_sub_test():
    config = OmegaConf.load('configs/dpt_sub04g18.yaml')
    print('config is:', config)
    model: torch.nn.Module = instantiate_from_config(config.model)
    model.eval()
    # out_folder = os.path.join(config.train.out_folder,
    #                           "%s_%d" % (config.train.out_name, config.train.exp_id))
    # weight = torch.load(os.path.join(out_folder, 'backbone.pth'))
    # model.load_state_dict(weight)
    # print('OPN Model loaded.')

    # print(model.transformer.layers[0][0].fn.to_qkv.weight)

    with torch.no_grad():
        from fvcore.nn import FlopCountAnalysis, parameter_count, parameter_count_table

        print("*" * 20, 'Full Model', "*" * 20)
        img = torch.randn(1, 3, 112, 112)
        flops = FlopCountAnalysis(model, img)
        # print("[fvcore] #Params Table\n", parameter_count_table(model, max_depth=7))
        print("[fvcore] FLOPs: %.2fG" % (flops.total() / 1e9))
        print("[fvcore] #Params: %.2fM" % (parameter_count(model)[''] / 1e6))

    # use_label = False
    # bs = 8
    # img = torch.randn(bs, 3, 112, 112).to(device)
    # label = torch.zeros((bs), dtype=torch.long).to(device)
    # model = model.to(device)
    # start = time.time()
    # for _ in tqdm(range(200)):
    #     out = model(img, label) if use_label else model(img)
    # print('Throughput: %.2f samples/sec' % (bs * 200 / (time.time() - start)))


def res_test():
    from backbone.resnet import resnet50
    from backbone.iresnet import iresnet50
    from backbone.spherenet import sphere

    dim = 128
    img_size = 112

    # model = resnet50(num_classes=dim)
    # model = iresnet50(num_features=dim)
    model = sphere(type=64)
    model.eval()
    print('Model loaded.')

    with torch.no_grad():
        from fvcore.nn import FlopCountAnalysis, parameter_count, parameter_count_table
        print("*" * 20, 'Full Model', "*" * 20)
        img = torch.randn(1, 3, img_size, img_size)
        flops = FlopCountAnalysis(model, img)
        # print("[fvcore] #Params Table\n", parameter_count_table(model, max_depth=7))
        print("[fvcore] FLOPs: %.2fG" % (flops.total() / 1e9))
        print("[fvcore] #Params: %.2fM" % (parameter_count(model)[''] / 1e6))
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[numel] #Params: %.2fM' % (params / 1e6))

    use_label = True
    bs = 8
    img = torch.randn(bs, 3, img_size, img_size).to(device)
    label = torch.zeros((bs), dtype=torch.long).to(device)
    model = model.to(device)
    # start = time.time()
    # for _ in tqdm(range(200)):
    #     out = model(img, label) if use_label else model(img)
    # print('Throughput: %.2f samples/sec' % (bs * 200 / (time.time() - start)))

    model.train()
    output = model(img, label) if use_label else model(img)
    output.mean().backward()
    print('Out shape:', output.shape)


if __name__ == "__main__":
    if torch.backends.mps.is_built():
        device = 'mps'
    elif torch.backends.cuda.is_built():
        device = 'cuda'
    else:
        device = 'cpu'
    print('Run on device: %s' % device)

    # dpt_sub_test()
    # opn_test()
    # ddp_test()
    # fet_test()
    fvit_test()
    # main()
    # res_test()
