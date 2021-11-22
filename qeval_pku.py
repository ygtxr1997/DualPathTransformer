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

divisor = 1
TASKS = {
    'pku_ver': {
        'img_root': '/home/yuange/dataset/PKU-Masked-Face',
        'list_file': 'ver24000.list',
        'save_path': './features',
        'task_name': 'pku_ver_extract.npy',
        'model_name': 'arcface_r18',
        'resume_path': '',
        'num_classes': 10575,
        'transform': transforms.Compose([transforms.Resize([112, 112]),
                                         # Randombaboon(),
                                         # Randomblock(),
                                         transforms.ToTensor()]),
        'ground_truth_label': list(np.zeros([3000 * 4 // divisor], dtype=np.int))
                              + list(np.ones([3000 * 4 // divisor], dtype=np.int)),
    },
}


class ExtractFeature(object):
    '''特征提取类'''
    def __init__(self, task):
        self.img_root = task['img_root']
        if not os.path.exists(self.img_root):
            self.img_root = '/GPUFS/sysu_zhenghch_1/yuange/datasets/' + self.img_root[len('/home/yuange/dataset/'):]
        self.list_file = task['list_file']
        self.save_path = task['save_path']
        self.task_name = task['task_name']
        self.model_name = task['model_name']
        self.resume_path = task['resume_path']
        self.num_classes = task['num_classes']
        self.transform = task['transform']

    def _load_model(self):
        if self.model_name == 'arcface_r18':
            # self.weight_path = '/home/yuange/code/SelfServer/ArcFace/r18-backbone.pth'
            # self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r18_occ6/backbone.pth'
            self.weight_path = '/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/recognition/arcface_torch/arcface_r18/backbone.pth'
            # self.weight_path = './tmp_47018/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('iresnet18'))(False).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r34':
            # self.weight_path = '/home/yuange/code/SelfServer/ArcFace/r34-backbone.pth'
            self.weight_path = './ms1mv3_arcface_r34_occ6/backbone.pth'
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
            weight = torch.load('/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r152_mg/backbone.pth')
            model = eval("backbone.{}".format('iresnet152'))(False).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'iresnetfc':
            # self.weight_path = '/home/yuange/code/SelfServer/ArcFace/r18-backbone.pth'
            # self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r18_occ6/backbone.pth'
            # self.weight_path = '/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/recognition/arcface_torch/arcface_r18_angle/backbone.pth'
            self.weight_path = './tmp_49018/backbone.pth'
            weight = torch.load(self.weight_path)
            model = eval("backbone.{}".format('iresnetfc18'))(False).cuda()
            model.load_state_dict(weight)
        elif self.model_name == 'arcface_r100_osb_r50':
            weight = torch.load('/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r100_osb_r50/backbone.pth')
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
            weight = torch.load('/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/resnet50/backbone.pth')
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

        return model

    def _load_one_input(self, img, index, flip=False, protocol='NB'):

        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if 'pku' in self.task_name:
            common_trans = transforms.Compose([transforms.Resize([112, 112]),
                                               transforms.ToTensor()])
        else:
            common_trans = transforms.Compose([transforms.CenterCrop([112, 112]),
                                               transforms.ToTensor()])

        if protocol == 'NB':
            img = self.transform(img) if index % 2 == 0 else common_trans(img)
        elif protocol == 'BB':
            img = self.transform(img)

        return img

    def _visualize_features(self, features):
        """ Visualize the 256-D features """
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        tsne.fit_transform(features)
        print(tsne.embedding_.shape)  # (12000, 2)

        embedding = tsne.embedding_
        min_val = embedding.min()
        max_val = embedding.max()
        heat_map = np.zeros((100, 100), dtype=np.uint8)
        for pairs in embedding:
            px = int((pairs[0] - min_val) / (max_val - min_val) * 98)
            py = int((pairs[1] - min_val) / (max_val - min_val) * 98)
            heat_map[px][py] += 1
        heat_map = ((heat_map / heat_map.max()) * 15).astype(np.uint8)

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.tick_params(labelsize=35)

        import seaborn as sns
        sns.heatmap(heat_map, xticklabels=20, yticklabels=20, cbar=None)

        save_name = 'features_' + self.task_name[:-12] + '.jpg'
        plt.savefig(os.path.join(self.save_path, save_name))
        plt.clf()

        self.heat_map = heat_map

    def _visualize_feature_map(self, feature_map, save_name):
        val_min, val_max = feature_map.min(), feature_map.max()  # [-1, 1]
        feature_map = feature_map.cpu().data.numpy()

        channel = feature_map.shape[1]
        h, w = feature_map.shape[-2], feature_map.shape[-1]

        heat_map = np.zeros((h, w), dtype=np.uint8)
        for b in range(1):
            for c in range(channel):
                for i in range(h):
                    for j in range(w):
                        heat_map[i][j] = int((feature_map[b][c][i][j] + 1.0) / 2 * 255)

                import matplotlib as mpl
                mpl.use('Agg')
                import matplotlib.pyplot as plt
                plt.figure(figsize=(5, 5))
                plt.tick_params(labelsize=35)
                plt.axis('off')

                import seaborn as sns
                sns.heatmap(heat_map, cbar=None, cmap='jet')

                save_path = os.path.join(self.save_path, 'latent')
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = os.path.join(save_path, str(b))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                save_path = os.path.join(save_path, str(h) + 'x' + str(w))
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                plt.savefig(os.path.join(save_path, str(c) + save_name), pad_inches=0, bbox_inches='tight', dpi=100)
                plt.clf()

    def _visualize_attn(self, mask, identity, in_image):
        # print(mask.shape)  # (32, 512, 7, 7)
        # print(identity.shape)  # (32, 512, 7, 7)

        # save input image
        for b in range(1):
            arr = in_image[b].cpu().data.numpy()
            rgb = (arr * 127 + 128)
            rgb_img = np.zeros([112, 112, 3])
            rgb_img[:, :, 0] = rgb[0]
            rgb_img[:, :, 1] = rgb[1]
            rgb_img[:, :, 2] = rgb[2]
            img = Image.fromarray(rgb_img.astype(np.uint8), mode='RGB')
            save_path = os.path.join(self.save_path, 'latent')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = os.path.join(save_path, str(b))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            img.save(os.path.join(save_path, 'input.jpg'))

        # visualize feature map
        self._visualize_feature_map(mask, 'attn.jpg')
        self._visualize_feature_map(identity, 'identity.jpg')
        self._visualize_feature_map(identity + identity * mask, 'cleaned.jpg')

    def start_extract(self, all_img, protocol='NB'):
        print("=> extract task started, task is '{}'".format(self.task_name))
        model = self._load_model()

        num = len(all_img)
        features = np.zeros((num, 512))
        features_flip = np.zeros((num, 512))

        # img to tensor
        all_input = torch.zeros(num, 3, 112, 112)
        for i in range(num):
            one_img = all_img[i]
            one_img_tensor = self._load_one_input(one_img, i, protocol=protocol)
            all_input[i, :, :, :] = one_img_tensor

        all_flip = torch.zeros(num, 3, 112, 112)
        for i in range(num):
            one_img = all_img[i]
            one_img_tensor = self._load_one_input(one_img, i, flip=True, protocol=protocol)
            all_flip[i, :, :, :] = one_img_tensor

        # start
        print("=> img-to-tensor is finished, start inference ...")
        # all_input = all_input.cuda()
        with torch.no_grad():
            all_input_var = torch.autograd.Variable(all_input)
            all_input_var = all_input_var.sub_(0.5).div_(0.5)  # [-1, 1]
            # print(all_input_var.min(), all_input_var.max())
            all_flip_var = torch.autograd.Variable(all_flip)
            all_flip_var = all_flip_var.sub_(0.5).div_(0.5)  # [-1, 1]

        batch_size = 32
        total_step = num // batch_size
        for i in range(total_step):
            patch_input = all_input_var[i * batch_size : (i + 1) * batch_size]
            # feature, mask, identity = model(patch_input)
            feature = model(patch_input.cuda())
            features[i * batch_size : (i + 1) * batch_size] = feature.data.cpu().numpy()

            # vis
            if i == -1:
                self._visualize_attn(mask, identity, patch_input)

        """ Visualize Predicted Masks """
        #     if i <= 400:
        #         # some_tensor.max(0)[0]: value of max_value
        #         # some_tensor.max(0)[1]: index of max_value
        #         mask = mask[0].cpu().max(0)[1].data.numpy() * 255
        #         mask = mask.astype(np.uint8)
        #
        #         mask = Image.fromarray(mask.astype(np.uint8))
        #         mask.save(os.path.join(self.save_path, 'pku' + str(i) + '_learned.jpg'))
        #
        #         img = np.zeros((112, 112, 3))
        #         img[:, :, 0] = (patch_input[0][0].cpu().data.numpy() + 1.0) * 127.5
        #         img[:, :, 1] = (patch_input[0][1].cpu().data.numpy() + 1.0) * 127.5
        #         img[:, :, 2] = (patch_input[0][2].cpu().data.numpy() + 1.0) * 127.5
        #
        #         img = Image.fromarray(img.astype(np.uint8), mode='RGB')
        #         img.save(os.path.join(self.save_path, 'pku' + str(i) + '_truth.jpg'))
        #
        # raise FileNotFoundError

        for i in range(total_step):
            patch_input = all_flip_var[i * batch_size: (i + 1) * batch_size]
            # feature, mask, identity = model(patch_input)
            feature = model(patch_input.cuda())
            features_flip[i * batch_size: (i + 1) * batch_size] = feature.data.cpu().numpy()

            # vis
            if i == -1:
                self._visualize_attn(mask, identity, patch_input)

        features = features_flip + features

        save_file = os.path.join(self.save_path, self.task_name)
        np.save(save_file, features)
        return features
        # print("=> extract task finished, file is saved at '{}'".format(save_file))

        # print("=> visualization started")
        # self._visualize_features(features)
        # print("=> visualization finished")

        # return ret_vector


class Verification(object):
    """人脸验证类"""
    def __init__(self, task):
        self.save_path = task['save_path']
        self.task_name = task['task_name']
        self.ground_truth_label = task['ground_truth_label']
        self._prepare()

    def _prepare(self):
        feature = np.load(os.path.join(self.save_path, self.task_name))
        feature = sklearn.preprocessing.normalize(feature)
        self.feature = feature

    def start_verification(self):
        # print("=> verification started, caculating ...")
        predict_label = []
        num = self.feature.shape[0]
        for i in range(num // 2):
            dis_cos = cdist(self.feature[i * 2: i * 2 + 1, :],
                            self.feature[i * 2 + 1: i * 2 + 2, :],
                            metric='cosine')
            predict_label.append(dis_cos[0, 0])

        fpr, tpr, threshold = roc_curve(self.ground_truth_label, predict_label)
        acc = tpr[np.argmin(np.abs(tpr - (1 - fpr)))]  # 选取合适的阈值
        print("=> verification finished, accuracy rate is {}".format(acc))
        ret_acc = acc

        roc_auc = auc(fpr, tpr)
        # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        plt.savefig(os.path.join(self.save_path, 'auc.jpg'))
        plt.clf()

        """ Calculate TAR@FAR<=1e-3 """
        neg_cnt = len(predict_label) // 2
        pos_cnt = neg_cnt
        self.ground_truth_label = np.array(self.ground_truth_label)
        predict_label = np.array(predict_label)
        pos_dist = predict_label[self.ground_truth_label == 0].tolist()
        neg_dist = predict_label[self.ground_truth_label == 1].tolist()

        far_val = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        ret_tarfar = []
        for idx in range(len(far_val)):

            threshold = []
            for T in neg_dist:
                neg_pair_smaller = 0.
                for i in range(neg_cnt):
                    if neg_dist[i] < T:
                        neg_pair_smaller += 1
                far = neg_pair_smaller / neg_cnt
                if far <= far_val[idx]:
                    threshold.append(T)

            acc = 0.
            print(len(threshold))
            for T in threshold:
                pos_pair_smaller = 0.
                for i in range(pos_cnt):
                    if pos_dist[i] <= T:
                        pos_pair_smaller += 1
                tar = pos_pair_smaller / pos_cnt
                acc = max(acc, tar)

            print("=> verification finished, accuracy rate (TAR@FAR<=1e-%d) is %.6f" % (idx + 1, acc))
            ret_tarfar.append(acc)

        return ret_acc, ret_tarfar


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch DualPathTransformer Training')
    parser.add_argument('--network', type=str, default='dpt_r18', help='backbone network')
    parser.add_argument('--dataset', type=str, default='pku_ver', help='pku_ver')
    args = parser.parse_args()

    random.seed(4)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    task_type = args.dataset
    my_task = TASKS[task_type]

    my_task['model_name'] = args.network
    my_task['task_name'] = my_task['task_name']
    print('[model_name]: ', my_task['model_name'])
    print('[transform]: ', my_task['transform'])

    """ Pre-load images into memory """
    print("=> Pre-loading images ...")
    img_root = my_task['img_root']
    if not os.path.exists(img_root):
        img_root = '/GPUFS/sysu_zhenghch_1/yuange/datasets/' + img_root[len('/home/yuange/dataset/'):]

    all_img = []
    pku_root = img_root
    pku_list_file = my_task['list_file']
    with open(os.path.join(pku_root, pku_list_file), 'r') as list_f:
        pku_list = list_f.readlines()
    pku_num = len(pku_list)
    num = pku_num
    pku_pos = []
    pku_neg = []
    for index in tqdm(range(pku_num)):
        img_path = os.path.join(pku_root, pku_list[index][:-4] + 'png')
        one_img = Image.open(img_path).convert('RGB')

        """ Crop Image (finished) """
        # h, w = one_img.size[1], one_img.size[0]
        # s = min(h, w)
        # pku_trans = transforms.Compose([transforms.CenterCrop([s, s]),
        #                                 transforms.Resize([112, 112])])
        # one_img = pku_trans(one_img)
        # one_img.save(img_path[:-3] + 'jpg')

        if index < 6000 * 4 // divisor:
            pku_pos.append(one_img)
        else:
            pku_neg.append(one_img)

    flag = False
    pos_cnt, neg_cnt = 0, 0
    for i in range(num):
        if i % (600 * 4 // divisor) == 0:
            flag = ~flag
        if flag:
            all_img.append(pku_pos[pos_cnt])
            pos_cnt += 1
        else:
            all_img.append(pku_neg[neg_cnt])
            neg_cnt += 1
    assert pos_cnt + neg_cnt == num

    """ Multi-Test """
    protocol = 'BB'
    lo_list = [0, ]
    hi_list = [0, ]
    # lo_list.reverse()
    # hi_list.reverse()
    avg_acc_list = []
    fars = [0., 0., 0., 0., 0.]
    weight_path = ''

    for ind in range(0, len(lo_list)):
        print('================== [ %d ] ===============' % ind)

        lo, hi = lo_list[ind], hi_list[ind]

        my_task['resume_path'] = ''
        my_task['save_path'] = 'features/'

        flag = bool(False)
        period = 300 * 4 // divisor
        issame_list = []
        for i in range(6000 * 4 // divisor):
            if i % period == 0:
                flag = ~flag
            issame_list.append(bool(flag))
        flag = 1
        intsame_list = []
        for i in range(6000 * 4 // divisor):
            if i % period == 0:
                flag = 1 - flag
            intsame_list.append(flag)
        my_task['ground_truth_label'] = intsame_list

        avg_acc = 0.
        repeat_time = 1 if (lo == 0 and hi == 0) or (lo == 100 and hi == 100) else 10
        for repeat in range(repeat_time):
            ExtractTask = ExtractFeature(my_task)
            features = ExtractTask.start_extract(all_img, protocol=protocol)

            weight_path = ExtractTask.weight_path

            features = sklearn.preprocessing.normalize(features)

            import eval.verification as ver
            _, _, accuracy, val, val_std, far = ver.evaluate(features, issame_list)
            acc2, std2 = np.mean(accuracy), np.std(accuracy)
            print('acc2 = [%.6f]' % acc2)
            avg_acc += acc2

            VerificationTask = Verification(my_task)
            _, fars = VerificationTask.start_verification()

        avg_acc = avg_acc / repeat_time

        avg_acc_list.append(avg_acc)
        print('[avg_acc]: %.4f' % (avg_acc))

    # print results
    print('[protocol]:', protocol)
    print('[model_name]:', my_task['model_name'])
    print('[weight_path]:', weight_path)
    for ind in range(0, len(avg_acc_list)):
        print('[%d ~ %d] | [avg_acc]: %.6f'
              % (lo_list[ind], hi_list[ind], avg_acc_list[ind]))
        print('          | [tar@far]: %.6f, %.6f, %.6f, %.6f, %.6f'
              % (fars[0], fars[1], fars[2], fars[3], fars[4]))