import os
import pickle

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cdist

import sys
import warnings

import torch

import backbone

from PIL import Image
import numpy as np
import random
import os
from torchvision import transforms
import time
from tqdm import tqdm
import re
from scipy.special import expit

from dataset.img_preprocess_no_msk import Randomblock, RandomConnectedPolygon, RandomTrueObject

from config import cfg

dataset_root = '/home/yuange/dataset'
if not os.path.exists(dataset_root):
    dataset_root = '/GPUFS/sysu_zhenghch_1/yuange/datasets'
assert os.path.exists(dataset_root)

TASKS = {
    'MegaFace': {
        'distractor_list': os.path.join(dataset_root, 'MegaFace/distractor.list'),
        'save_path': 'features',
        'task_name': 'megaface.npy',
        'distractor_npy': 'megaface_distractor_extract.npy',
        'probe_path': os.path.join(dataset_root, 'MegaFace/facescrub_images'),
        'probe_list': os.path.join(dataset_root, 'MegaFace/probe.list'),
        'probe_npy': 'megaface_probe_extract.npy',
        # ------------------------------------------------------------------------
        'model_name': 'mskfuse_light_y',
        'resume_path': '',
        'num_classes': 10575,
        'transform': transforms.Compose([transforms.Resize([112, 112]),
                                         RandomTrueObject('resources/occ/object_test'),
                                         transforms.ToTensor()]),
        'distractor_label': 'distractor_label.npy',
        'probe_label': 'probe_label.npy'
    },
}


class ExtractFeature(object):
    """特征提取类"""

    def __init__(self, task):
        self.distractor_list = task['distractor_list']
        self.save_path = task['save_path']
        self.distractor_npy = task['distractor_npy']
        self.probe_path = task['probe_path']
        self.probe_list = task['probe_list']
        self.probe_npy = task['probe_npy']

        self.model_name = task['model_name']
        self.resume_path = task['resume_path']
        self.num_classes = task['num_classes']
        self.transform = task['transform']

        self.distractor_label = task['distractor_label']
        self.probe_label = task['probe_label']

        self.feats_probe = {}
        self.labels_probe = {}

    def _load_model(self):
        if self.model_name == 'arcface_r18':
            # self.weight_path = '/home/yuange/code/SelfServer/ArcFace/r18-backbone.pth'
            # self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r18_occ6/backbone.pth'
            self.weight_path = '/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/recognition/arcface_torch/arcface_r18_angle/backbone.pth'
            # self.weight_path = './tmp_47018/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('iresnet18'))(False).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r34':
            # self.weight_path = '/home/yuange/code/SelfServer/ArcFace/r34-backbone.pth'
            self.weight_path = './arcface_r34_7occ/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('iresnet34'))(False).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r50':
            # self.weight_path = '/home/yuange/code/SelfServer/ArcFace/r50-backbone.pth'
            self.weight_path = 'ms1mv3_arcface_r50_occ6/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('iresnet50'))(False).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r100':
            # self.weight_path = '/home/yuange/code/SelfServer/ArcFace/r100-backbone.pth'
            # self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r100_onlysmooth/backbone.pth'
            self.weight_path = 'ms1mv3_arcface_r100_mg/backbone.pth'
            # weight = torch.load('/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r100_bot50_lr3/backbone.pth')
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('iresnet100'))(False).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r152':
            # weight = torch.load('/home/yuange/code/SelfServer/ArcFace/r152-backbone.pth')
            weight = torch.load(
                '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r152_mg/backbone.pth')
            model = eval("backbone.{}".format('iresnet152'))(False).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r100_osb_r50':
            weight = torch.load(
                '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r100_osb_r50/backbone.pth')
            model = eval("{}".format('iresnet100_osb'))(False,
                                                        dropout=0,
                                                        fp16=False,
                                                        osb='r50').cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r50_osb_r18':
            self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r50_osb_r18_aaai/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("{}".format('iresnet50_osb'))(False,
                                                       dropout=0,
                                                       fp16=False,
                                                       osb='r18').cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r50_osb_r34':
            self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r50_osb_r34_aaai/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("{}".format('iresnet50_osb'))(False,
                                                       dropout=0,
                                                       fp16=False,
                                                       osb='r34').cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r50_osb_r50':
            self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r50_osb_r50_seg01_aaai/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("{}".format('iresnet50_osb'))(False,
                                                       dropout=0,
                                                       fp16=False,
                                                       osb='r50').cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r18_osb_r18':
            self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r18_osb_r18_aaai/backbone.pth'
            # self.weight_path = '/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r18_osb18_mlm4_1115_drop01_swinmei/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("{}".format('iresnet18_osb'))(False,
                                                       dropout=0,
                                                       fp16=False,
                                                       osb='r18',
                                                       mlm_type=cfg.mlm_type,
                                                       mlm_list=cfg.mlm_list,
                                                       ).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r18_osb_r34':
            self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r18_osb_r34_aaai/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("{}".format('iresnet18_osb'))(False,
                                                       dropout=0,
                                                       fp16=False,
                                                       osb='r34').cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r18_osb_r50':
            self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r18_osb_r50_aaai/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("{}".format('iresnet18_osb'))(False,
                                                       dropout=0,
                                                       fp16=False,
                                                       osb='r50').cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r34_osb_r18':
            self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r34_osb_r18_aaai/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("{}".format('iresnet34_osb'))(False,
                                                       dropout=0,
                                                       fp16=False,
                                                       osb='r18').cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r34_osb_r50':
            self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r34_osb_r50_aaai/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("{}".format('iresnet34_osb'))(False,
                                                       dropout=0,
                                                       fp16=False,
                                                       osb='r50').cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r50_res':
            weight = torch.load(
                '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/resnet50/backbone.pth')
            model = eval("backbone.{}".format('resnet50'))(False).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r100_osb_r18':
            self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r100_osb_r18_aaai/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("{}".format('iresnet100_osb'))(False,
                                                        dropout=0,
                                                        fp16=False,
                                                        osb='r18').cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'ft_r18':
            self.weight_path = './tmp_33102/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('ft_r18'))(False,
                                                         fp16=cfg.fp16,
                                                         num_classes=cfg.num_classes,
                                                         dim=cfg.model_set.dim,
                                                         depth=cfg.model_set.depth,
                                                         heads=cfg.model_set.heads,
                                                         mlp_dim=cfg.model_set.mlp_dim,
                                                         emb_dropout=cfg.model_set.emb_dropout,
                                                         dim_head=cfg.model_set.dim_head,
                                                         dropout=cfg.model_set.dropout
                                                         ).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'ft_r50':
            self.weight_path = './tmp_35106/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('ft_r50'))(False,
                                                         fp16=cfg.fp16,
                                                         num_classes=cfg.num_classes,
                                                         dim=384,
                                                         depth=2,
                                                         heads=8,
                                                         mlp_dim=256,
                                                         emb_dropout=0.,
                                                         dim_head=64,
                                                         dropout=0.
                                                         ).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'dpt_r18':
            self.weight_path = './tmp_0/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('dpt_r34s3_ca1'))(False,
                                                                fp16=cfg.fp16,
                                                                num_classes=cfg.num_classes,
                                                                dim=cfg.model_set.dim,
                                                                depth=cfg.model_set.depth,
                                                                heads_id=cfg.model_set.heads_id,
                                                                heads_oc=cfg.model_set.heads_oc,
                                                                mlp_dim_id=cfg.model_set.mlp_dim_id,
                                                                mlp_dim_oc=cfg.model_set.mlp_dim_oc,
                                                                emb_dropout=cfg.model_set.emb_dropout,
                                                                dim_head_id=cfg.model_set.dim_head_id,
                                                                dim_head_oc=cfg.model_set.dim_head_oc,
                                                                dropout_id=cfg.model_set.dropout_id,
                                                                dropout_oc=cfg.model_set.dropout_oc
                                                                ).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'dptsa_r18':
            self.weight_path = './tmp_42124/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('dpt_only_sa_r34'))(False,
                                                                  fp16=cfg.fp16,
                                                                  num_classes=cfg.num_classes,
                                                                  dim=cfg.model_set.dim,
                                                                  depth=cfg.model_set.depth,
                                                                  heads=cfg.model_set.heads,
                                                                  mlp_dim=cfg.model_set.mlp_dim,
                                                                  emb_dropout=cfg.model_set.emb_dropout,
                                                                  dim_head=cfg.model_set.dim_head,
                                                                  dropout=cfg.model_set.dropout,
                                                                  ).cuda()
            model.load_state_dict(weight)
        else:
            print('Error model type\n')

        model.eval()
        model = torch.nn.DataParallel(model).cuda()

        if self.resume_path:
            print("=> loading checkpoint '{}'".format(self.resume_path))
            checkpoint = torch.load(self.resume_path)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(self.resume_path))

        self.model = model
        return model

    def start_extract(self):
        print("=> extract task started, task is '{}'".format(self.distractor_list))
        model = self._load_model()

        # Stage 1/2: Save distractor feature
        features = []
        labels = []

        print("=> extract distractors features:")

        """-------------- test_loader ---------------------"""
        batch_size = 128
        test_loader = torch.utils.data.DataLoader(
            ImageList(root='', fileList=self.distractor_list,
                      transform=self.transform, is_train=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=6,
            pin_memory=True
        )

        sum_cnt = 1000000
        features = np.zeros((sum_cnt, 512))
        labels = np.zeros((sum_cnt))
        sum_cnt = 0

        start = time.time()
        for i, (input, target) in enumerate(test_loader):

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            from torch.autograd.variable import Variable
            with torch.no_grad():
                input_var, target_var = Variable(input), Variable(target)

            # output, seg = model(input_var)
            output = model(input_var)

            output = output.data.cpu().numpy().astype(np.float32)
            target_var = target_var.data.cpu().numpy()

            offset = output.shape[0]
            features[i * batch_size : i * batch_size + offset, :] = output
            labels[i * batch_size : i * batch_size + offset] = target_var
            sum_cnt += offset

            if i % 100 == 0:
                print('[{0}/{1}]:\t'
                      'Time: {2}\t'
                      'Features shape: {3}\t'
                      'Labels shape: {4}'.format(i, len(test_loader), int(time.time() - start),
                      features.shape, labels.shape),
                      end='\r',
                      flush=True
                      )
                start = time.time()

        print(self.save_path)
        print(self.distractor_npy)
        save_distractor_file = os.path.join(self.save_path, self.distractor_npy)
        np.save(save_distractor_file, features)
        np.save(os.path.join(self.save_path, self.distractor_label), labels)
        print('sum_cnt is %d' % sum_cnt)

        # Stage 2/2: Save probe feature
        cur_id = 0

        print("=> extract probe features:")
        print("=> do nothing.")
        print("=> extract task finished, file is saved at '{}'".format(self.save_path))

    def _prepare(self):
        # Get features of gallery and test
        feats_distractor = np.load(os.path.join(self.save_path, self.distractor_npy))
        feats_distractor = sklearn.preprocessing.normalize(feats_distractor)
        self.feats_distractor = feats_distractor

        # Get ground truth id of distractors
        self.labels_distractor = np.load(os.path.join(self.save_path, self.distractor_label))

    def start_identification(self):

        model = self.model
        self._prepare()

        print("=> identification started, caculating ...")

        right, wrong = 0, 0
        cur_id = 0

        out_path = os.path.join(self.save_path, 'mega')
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        for identity in tqdm(os.listdir(self.probe_path)):

            cur_path = os.path.join(self.probe_path, identity)
            cur_id += 1

            features_probe_ori = []
            features_probe_occ = []

            # Extract all probe images features of id = identity
            img_cnt = 0
            for img in os.listdir(cur_path):

                img = Image.open(os.path.join(cur_path, img)).convert('RGB')

                trans_ori = transforms.Compose([transforms.Resize([112, 112]), transforms.ToTensor()])
                img_ori = trans_ori(img)
                img_occ = self.transform(img)

                one_input_ori = torch.zeros(1, 3, 112, 112)
                one_input_ori[0, :, :, :] = img_ori
                one_input_occ = torch.zeros(1, 3, 112, 112)
                one_input_occ[0, :, :, :] = img_occ

                one_input_ori = one_input_ori.cuda()
                one_input_occ = one_input_occ.cuda()
                with torch.no_grad():
                    one_input_ori_var = torch.autograd.Variable(one_input_ori)
                    one_input_occ_var = torch.autograd.Variable(one_input_occ)
                    one_input_ori_var = one_input_ori_var.sub_(0.5).div_(0.5)  # [-1, 1]
                    one_input_occ_var = one_input_occ_var.sub_(0.5).div_(0.5)  # [-1, 1]

                # feature_ori, _ = model(one_input_ori_var)
                # feature_occ, seg = model(one_input_occ_var)
                feature_ori = model(one_input_ori_var)
                feature_occ = model(one_input_occ_var)

                output_ori = feature_ori.data.cpu().numpy().astype(np.float32)
                output_occ = feature_occ.data.cpu().numpy().astype(np.float32)

                if len(features_probe_ori) == 0:
                    features_probe_ori = output_ori
                    features_probe_occ = output_occ
                else:
                    features_probe_ori = np.append(features_probe_ori, output_ori, axis=0)
                    features_probe_occ = np.append(features_probe_occ, output_occ, axis=0)

                img_cnt += 1
                # if img_cnt % 30 == 0 or True:
                #     # save snapshot
                #     snapshot = np.zeros((112, 112, 3), dtype=np.uint8)
                #     snapshot[:, :, 0] = (one_input_occ[0][0].cpu().data.numpy() + 1.0) * 127.5
                #     snapshot[:, :, 1] = (one_input_occ[0][1].cpu().data.numpy() + 1.0) * 127.5
                #     snapshot[:, :, 2] = (one_input_occ[0][2].cpu().data.numpy() + 1.0) * 127.5
                #     snapshot = Image.fromarray(snapshot.astype(np.uint8))
                #     snapshot.save(os.path.join(out_path, identity + str(img_cnt) + '_gt.jpg'))
                #
                #     mask = seg[0].cpu().max(0)[1].data.numpy() * 255
                #     mask = Image.fromarray(mask.astype(np.uint8))
                #     mask.save(os.path.join(out_path, identity + str(img_cnt) + '_learned.jpg'))

            # normalization
            norm_f = np.sum(features_probe_ori ** 2, axis=1, keepdims=True)
            features_probe_ori = features_probe_ori / norm_f

            norm_f = np.sum(features_probe_occ ** 2, axis=1, keepdims=True)
            features_probe_occ = features_probe_occ / norm_f

            dists_img1_distractor = cdist(features_probe_occ, self.feats_distractor,  'cosine')

            min_dists_img1_distractor = np.min(dists_img1_distractor, axis=1)

            dists_img1_img2 = cdist(features_probe_occ, features_probe_ori, 'cosine')

            for img1 in range(dists_img1_img2.shape[0]):

                for img2 in range(dists_img1_img2.shape[1]):

                    if img1 == img2:
                        continue

                    if min_dists_img1_distractor[img1] < dists_img1_img2[img1][img2]:
                        wrong += 1
                    else:
                        right += 1

        print("=> identification finished, accuracy rate is {}".format(right / (right + wrong)))


def megaface_loader(path):
    img = Image.open(path).convert('RGB')
    return img


def megaface_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList


from torch.utils import data
class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=megaface_list_reader, loader=megaface_loader,
                 is_train=False):
        self.root      = root
        self.imgList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader
        self.is_train  = is_train

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.is_train:
            img = img.resize([112, 112])
        else:
            img = img.resize([112, 112])

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)


def vis_segs(segs, save_path):
    print("==> start visualize segs ...")
    print(segs.shape)
    for ind in range(segs.shape[0]):
        seg = segs[ind]
        mask = seg
        mask = Image.fromarray(mask.astype(np.uint8))

        save_name = 'seg_' + str(ind) + '.jpg'
        mask.save(os.path.join(save_path, save_name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch DualPathTransformer Training')
    parser.add_argument('--network', type=str, default='dpt_r18', help='backbone network')
    parser.add_argument('--dataset', type=str, default='MegaFace', help='MegaFace')
    args = parser.parse_args()

    import random
    random.seed(4)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    my_task = TASKS['MegaFace']
    print(my_task['transform'])

    my_task['model_name'] = args.network
    my_task['task_name'] = my_task['task_name'][:-4] + str(args.network) + '.npy'
    print('[transform]: ', my_task['transform'])
    my_task['distractor_label'] = 'distractor_label' + str(args.network) + '.npy'
    my_task['probe_label'] = 'probe_label.npy' + str(args.network) + '.npy'
    my_task['distractor_npy'] = 'megaface_distractor_extract' + str(args.network) + '.npy'

    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # task_list = ['arcface_r18',
    #              'arcface_r18_osb_r18',
    #              'arcface_r18_osb_r34',
    #              'arcface_r18_osb_r50']
    task_list = [args.network]

    # 多次测试
    for ind in range(len(task_list)):
        print('===================== object-occ [%d] ======================' % ind)

        my_task['resume_path'] = ''
        my_task['save_path'] = 'features/'
        my_task['model_name'] = task_list[ind]
        my_task['transform'] = transforms.Compose([transforms.Resize([112, 112]),
                                                   RandomTrueObject('resources/occ/object_test'),
                                                   Randomblock(1, 40),
                                                   transforms.ToTensor()])
        print('[model_name]: ', my_task['model_name'])

        ExtractTask = ExtractFeature(my_task)
        ExtractTask.start_extract()
        ExtractTask.start_identification()

    for ind in range(len(task_list)):
        print('===================== no-occ [%d] ======================' % ind)

        my_task['resume_path'] = ''
        my_task['save_path'] = 'features/'
        my_task['model_name'] = task_list[ind]
        my_task['transform'] = transforms.Compose([transforms.Resize([112, 112]),
                                                   transforms.ToTensor()])
        print('[model_name]: ', my_task['model_name'])

        ExtractTask = ExtractFeature(my_task)
        ExtractTask.start_extract()
        ExtractTask.start_identification()
