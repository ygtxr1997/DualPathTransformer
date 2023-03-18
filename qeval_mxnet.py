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
from torch.utils.data import DataLoader
from torchvision.transforms.functional import hflip

import backbones

from PIL import Image
import numpy as np
import random
import os
from torchvision import transforms
import time
from tqdm import tqdm
import re
from scipy.special import expit
from typing import Union

from datasets.augment.rand_occ import RandomBlock, RandomConnectedPolygon, RandomRealObject
from eval.eval_dataset import EvalDataset, MXNetEvalDataset, MXNetTsneDataset, MFR2EvalDataset, MFVEvalDataset
from eval.eval_dataset import MXNetRestorerDataset
from eval.eval_dataset import IdentifyDataset, MegaFaceDataset

from config import config_init, load_yaml


class MXNetEvaluator(object):
    def __init__(self,
                 eval_dataset: Union[EvalDataset, IdentifyDataset],
                 occ_trans,
                 cfg,
                 args,
                 ):
        """ MXNetEvaluator """
        ''' yaml config '''
        self.cfg = cfg
        self.is_gray = cfg.is_gray
        self.channel = 1 if self.is_gray else 3
        self.out_size = cfg.out_size  # (w,h)
        self.dim_feature = cfg.dim_feature

        ''' dataset & dataloader '''
        self.eval_mode = '1:1' if isinstance(eval_dataset, EvalDataset) else '1:N'
        self.eval_dataset = eval_dataset
        self.eval_dataset.update_occ_trans(occ_trans)
        for bs in np.arange(40, 0, -1):
            if len(eval_dataset) % bs == 0:
                self.batch_size = int(bs)
                print('[Batch Size]: %d' % bs)
                break
        self.eval_loader = DataLoader(
            self.eval_dataset, self.batch_size,
            num_workers=12, shuffle=False, drop_last=False
        )

        if self.eval_mode == '1:1':
            issame_list = eval_dataset.issame_list
            self.num = len(eval_dataset) * 2  # n = pair * 2
            self.issame_list = issame_list  # True:same
            self.intsame_list = [0 if x else 1 for x in issame_list]  # 0:same
        else:
            self.num = len(eval_dataset)
            self.probe_cnt = len(eval_dataset.probe_img)
            self.gallery_cnt = len(eval_dataset.gallery_img)

        ''' args '''
        self.model_name = args.network
        self.weight_folder = args.weight_folder
        self.is_vis = args.is_vis
        self.dataset_name = args.dataset
        self.tsne_id_cnt = args.identity_cnt  # used by tsne

        ''' model '''
        self.model = self._load_model()

        ''' visualization '''
        save_folder = os.path.join('./vis', self.dataset_name)
        if os.path.exists(save_folder):
            os.system('rm -r %s' % save_folder)
        os.makedirs(save_folder, exist_ok=True)
        self.save_folder = save_folder

    def _load_model(self):
        cfg = self.cfg
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
        elif self.model_name == 'msml':
            # self.weight_path = '/home/yuange/code/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r18_osb_r18_aaai/backbone.pth'
            # self.weight_path = '/GPUFS/sysu_zhenghch_1/yuange/SelfServer/DeepInsight/insightface/recognition/arcface_torch/ms1mv3_arcface_r18_osb18_mlm4_1115_drop01_swinmei/backbone.pth'
            self.weight_path = os.path.join(self.weight_folder, 'backbone.pth')
            weight = torch.load(self.weight_path)
            model = eval("backbones.{}".format('MSML'))(frb_type=cfg.frb_type,
                                                        frb_pretrained=False,
                                                        osb_type=cfg.osb_type,
                                                        fm_layers=cfg.fm_layers,
                                                        header_type=cfg.header_type,
                                                        header_params=cfg.header_params,
                                                        num_classes=cfg.num_classes,
                                                        fp16=False,
                                                        use_osb=cfg.use_osb,
                                                        fm_params=cfg.fm_params,
                                                        peer_params=cfg.peer_params,
                                                        ).cuda()
            model.load_state_dict(weight)
            for fm_idx in range(4):
                fm_op = model.frb.fm_ops[fm_idx]
                fm_op.open_saving(self.is_vis)
        elif 'from2021' in self.model_name:
            self.weight_path = ''
            print('loading TPAMI2021 FROM model...')
            model = backbones.From2021()
        else:
            raise ValueError('Error model type\n')

        model.eval()
        model = torch.nn.DataParallel(model).cuda()

        return model

    def _infer(self, x, ori=None):
        if ori is None:
            output = self.model(x)
        else:
            output = self.model(x, ori=ori)

        if type(output) is tuple:
            feature = output[0]
            final_seg = output[1]
        else:
            feature = output
            final_seg = None

        return feature, final_seg

    def _vis_segmentation_result(self,
                                 img: torch.Tensor,
                                 seg: torch.Tensor,
                                 index_list: list):
        print('Visualizing segmentation results...')
        save_folder = self.save_folder
        n = self.num
        assert len(index_list) * 2 <= n
        for idx in range(n // 2):
            if not idx in index_list:
                continue
            ''' predicted segmentation masks '''
            seg1_pil, seg2_pil = self.__t2p_segmentation_result(seg[idx * 2], seg[idx * 2 + 1])
            seg1_pil.save(os.path.join(save_folder, str(idx * 2) + '_predict.jpg'))
            seg2_pil.save(os.path.join(save_folder, str(idx * 2 + 1) + '_predict.jpg'))

            ''' input faces '''
            img1_pil, img2_pil = self.__t2p_input(img[idx * 2], img[idx * 2 + 1], is_gray=self.is_gray)
            img1_pil.save(os.path.join(save_folder, str(idx * 2) + '_input.jpg'))
            img2_pil.save(os.path.join(save_folder, str(idx * 2 + 1) + '_input.jpg'))

    @staticmethod
    def __t2p_segmentation_result(seg1: torch.Tensor, seg2: torch.Tensor):
        assert seg1.ndim == 3  # (C,H,W)
        ''' predicted segmentation masks '''
        seg1_np = seg1.max(0)[1].cpu().numpy() * 255  # (h,w)
        seg2_np = seg2.max(0)[1].cpu().numpy() * 255  # (h,w)
        seg1_pil = Image.fromarray(seg1_np.astype(np.uint8))
        seg2_pil = Image.fromarray(seg2_np.astype(np.uint8))
        return seg1_pil, seg2_pil

    @staticmethod
    def __t2p_input(img1: torch.Tensor, img2: torch.Tensor, is_gray: bool = False):
        assert img1.ndim == 3  # (C,H,W)
        ''' input faces '''
        if is_gray:
            img1_pil = Image.fromarray((img1.cpu().numpy() * 255).astype(np.uint8), mode='L')
            img2_pil = Image.fromarray((img2.cpu().numpy() * 255).astype(np.uint8), mode='L')
        else:
            img1_pil = Image.fromarray(((img1.permute(1, 2, 0).cpu().numpy() + 1.) * 127.5).astype(np.uint8),
                                       mode='RGB')
            img2_pil = Image.fromarray(((img2.permute(1, 2, 0).cpu().numpy() + 1.) * 127.5).astype(np.uint8),
                                       mode='RGB')
        return img1_pil, img2_pil

    def _vis_intermediate_feature_maps(self,
                                       index_list: list):
        """ Visualize Intermediate Feature Maps of FM Operators """
        print('Visualizing intermediate feature maps...')
        save_folder = self.save_folder
        n = self.num
        assert len(index_list) <= n

        for fm_idx in (0, 1, ):
            fm_op = self.model.module.frb.fm_ops[fm_idx]
            fm_op.plot_intermediate_feature_maps(vis_index_list=index_list,
                                                 save_folder=save_folder)

    def _vis_restored_feature_maps(self,
                                   index_list: list):
        """ Visualize Restored Feature Maps of FM Operators """
        print('Visualizing restored feature maps...')
        save_folder = self.save_folder
        n = self.num
        assert len(index_list) <= n

        for fm_idx in (0, 1, 2, 3):
            fm_op = self.model.module.frb.fm_ops[fm_idx]
            fm_op.plot_restored_feature_maps(vis_index_list=index_list,
                                             save_folder=save_folder)

    @staticmethod
    def __t2n_feature_map(feature_map: torch.Tensor):
        assert feature_map.ndim == 3  # (C,H,W), in [-1,1]
        c, h, w = feature_map.shape
        feature_map_nps = []
        for channel in range(c):
            one_channel = feature_map[c]
            one_channel = one_channel.permute(1, 2, 0).cpu().numpy()
            one_channel = (one_channel - one_channel.min()) / (one_channel.max() - one_channel.min())  # (H,W), in [0,1]
            feature_map_nps.append(one_channel)
        return feature_map_nps

    def _vis_intermediate_feature_pairs(self,
                                        seg_batches: torch.Tensor,
                                        index_list: list):
        """ Visualize Intermediate Feature Pairs of FM Operators """
        print('Visualizing intermediate feature pairs...')
        B, C, H, W = seg_batches.shape  # batched order
        save_folder = self.save_folder
        n = self.num
        assert len(index_list) <= n

        mask = torch.zeros((B, H, W))
        for b in range(B):
            mask[b] = seg_batches[b].max(0)[1]
        mask = mask.data  # 0-occ, 1-clean

        for fm_idx in range(4):
            fm_op = self.model.module.frb.fm_ops[fm_idx]
            fm_op.plot_intermediate_feature_pairs(gt_occ_msk=mask,
                                                  vis_index_list=index_list,
                                                  save_folder=save_folder)

    def _vis_embeddings_tsne(self,
                             embeddings: np.ndarray,
                             index_list: list = None,
                             ):
        if 'tsne' not in self.dataset_name:
            return
        print('Visualizing embeddings tsne...')
        from sklearn.manifold import TSNE
        from utils.vis_tsne import vis_tsne2d

        if index_list is None:
            index_list = np.arange(embeddings.shape[0])
        save_folder = self.save_folder
        embeddings = embeddings[index_list].astype(np.float)
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        tsne.fit_transform(embeddings)  # (N,2)

        tsne_2d: np.ndarray = tsne.embedding_
        save_name = os.path.join(save_folder, 'tsne.jpg')
        vis_tsne2d(tsne_2d, save_name, identity_cnt=self.tsne_id_cnt)

    def _vis_embeddings_heat(self,
                             embeddings: np.ndarray,
                             index_list: list = None):
        print('Visualizing embeddings heat...')
        save_folder = self.save_folder
        from eval.vis_heat import vis_feature_1d
        vis_list = []
        for index in index_list:
            vis_list.append(index * 2)
            vis_list.append(index * 2 + 1)
        embeddings = embeddings[vis_list].astype(np.float)
        for idx, embedding in enumerate(embeddings):
            save_name = os.path.join(save_folder, '%d_heat.jpg' % idx)
            vis_feature_1d(embedding, flatten_w=32, save_name=save_name)

    @torch.no_grad()
    def _1to1_verification(self):
        w, h = self.out_size
        n = self.num  # n = pair * 2 = (batch * batch_size) * 2
        dim_feature = self.dim_feature
        features = np.zeros((n, dim_feature))  # [feat1,...,feat1,feat2,...,feat2]
        features_flip = np.zeros_like(features)
        seg_batches = torch.zeros((n, 2, h, w))
        seg_s = torch.zeros_like(seg_batches)
        if self.is_vis:
            max_vis_cnt = 400
            img_batches = torch.zeros((n, self.channel, h, w))  # [img1_batch,img2_batch,...,img1_batch,img2_batch]
            img_s = torch.zeros_like(img_batches)  # [img1,...,img1,img2,...,img2]

        ''' 1. Extract Features '''
        print("=> start inference ...")
        batch_idx = 0
        for batch in tqdm(self.eval_loader):
            if batch_idx * self.batch_size >= 400 and self.is_vis:
                continue
            img1, img2, same = batch
            img_ori = None

            ''' a. original input '''
            img1 = img1.cuda()
            img2 = img2.cuda()
            if 'restore' in self.dataset_name:
                img_ori = img2

            feat1, seg1 = self._infer(img1, ori=img_ori)
            feat2, seg2 = self._infer(img2, ori=img_ori)

            features[batch_idx * self.batch_size:
                     (batch_idx + 1) * self.batch_size] = feat1.cpu().numpy()
            features[batch_idx * self.batch_size + (n // 2):
                     (batch_idx + 1) * self.batch_size + (n // 2)] = feat2.cpu().numpy()

            if self.is_vis:
                img1_cpu = img1.cpu()
                img2_cpu = img2.cpu()
                img_s[batch_idx * self.batch_size:
                      (batch_idx + 1) * self.batch_size] = img1_cpu
                img_s[batch_idx * self.batch_size + (n // 2):
                      (batch_idx + 1) * self.batch_size + (n // 2)] = img2_cpu
                img_batches[batch_idx * self.batch_size * 2:
                            batch_idx * self.batch_size * 2 + self.batch_size] = img1_cpu
                img_batches[batch_idx * self.batch_size * 2 + self.batch_size:
                            batch_idx * self.batch_size * 2 + self.batch_size * 2] = img2_cpu

            if seg1 is not None:
                seg1_cpu = seg1.cpu()
                seg2_cpu = seg2.cpu()
                seg_s[batch_idx * self.batch_size:
                      (batch_idx + 1) * self.batch_size] = seg1_cpu
                seg_s[batch_idx * self.batch_size + (n // 2):
                      (batch_idx + 1) * self.batch_size + (n // 2)] = seg2_cpu
                seg_batches[batch_idx * self.batch_size * 2:
                            batch_idx * self.batch_size * 2 + self.batch_size] = seg1_cpu
                seg_batches[batch_idx * self.batch_size * 2 + self.batch_size:
                            batch_idx * self.batch_size * 2 + self.batch_size * 2] = seg2_cpu

            ''' b. flipped input '''
            img1_flip = hflip(img1)
            img2_flip = hflip(img2)

            feat1_flip, _ = self._infer(img1_flip)
            feat2_flip, _ = self._infer(img2_flip)
            features_flip[batch_idx * self.batch_size:
                          (batch_idx + 1) * self.batch_size] = feat1_flip.cpu().numpy()
            features_flip[batch_idx * self.batch_size + (n // 2):
                          (batch_idx + 1) * self.batch_size + (n // 2)] = feat2_flip.cpu().numpy()

            batch_idx += 1

        features = features_flip + features

        ''' 2. Calculate Metrics '''
        predict_label = []
        features_reorder = np.zeros_like(features)  # [feat1,feat2,feat1,feat2,...,feat1,feat2]
        seg_reorder = torch.zeros_like(seg_s)
        if self.is_vis:
            img_reorder = torch.zeros_like(img_s)
        for i in range(n // 2):
            feat1 = features[i: i + 1, :]
            feat2 = features[i + (n // 2): i + 1 + (n // 2), :]
            dis_cos = cdist(feat1,
                            feat2,
                            metric='cosine')
            features_reorder[i * 2] = features[i]
            features_reorder[i * 2 + 1] = features[i + (n // 2)]
            seg_reorder[i * 2] = seg_s[i]
            seg_reorder[i * 2 + 1] = seg_s[i + (n // 2)]
            if self.is_vis:
                img_reorder[i * 2] = img_s[i]
                img_reorder[i * 2 + 1] = img_s[i + (n // 2)]
            predict_label.append(dis_cos[0, 0])

        """ (0) Visualization """
        if self.is_vis:
            vis_index = np.arange(min(100, n)).tolist()
            # self._vis_segmentation_result(img_reorder, seg_reorder, index_list=vis_index)
            # self._vis_embeddings_heat(features_reorder, index_list=vis_index)
            # self._vis_embeddings_tsne(features, index_list=np.arange(min(200, n // 2)))
            # self._vis_intermediate_feature_pairs(seg_batches, index_list=[0, 1, 2, 3])  # not work
            # self._vis_intermediate_feature_maps(index_list=[0])
            self._vis_restored_feature_maps(index_list=[0])
            print('Visualization finished, exiting.')
            exit()

        """ (1) Calculate Accuracy """
        fpr, tpr, threshold = roc_curve(self.intsame_list, predict_label)
        acc = tpr[np.argmin(np.abs(tpr - (1 - fpr)))]  # choose proper threshold
        print("=> verification finished, accuracy rate is {}".format(acc))
        ret_acc = acc

        # plot auc curve
        # roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        # plt.savefig(os.path.join(self.save_path, 'auc.jpg'))
        # plt.clf()

        if self.dataset_name not in ('mfr2', 'mfv',):
            import eval.verification as ver
            features_normed = sklearn.preprocessing.normalize(features_reorder)
            _, _, accuracy, val, val_std, far = ver.evaluate(features_normed, self.issame_list)
            acc2, std2 = np.mean(accuracy), np.std(accuracy)
            print('acc2 = [%.6f]' % acc2)
            ret_acc = acc2

        """ (2) Calculate TAR@FAR<=1e-k """
        neg_cnt = len(predict_label) // 2
        pos_cnt = neg_cnt
        ground_truth_label = np.array(self.intsame_list)
        predict_label = np.array(predict_label)
        pos_dist = predict_label[ground_truth_label == 0].tolist()
        neg_dist = predict_label[ground_truth_label == 1].tolist()

        far_val = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        ret_tarfar = np.zeros((len(far_val)))
        for idx in range(len(far_val)):

            """ Choose the far values """
            if idx > 3:
                continue

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
            ret_tarfar[idx] = acc

        return ret_acc, ret_tarfar

    @torch.no_grad()
    def _1ton_identification(self):
        w, h = self.out_size
        n = self.num  # n = probe * 2 + gallery
        dim_feature = self.dim_feature
        features = np.zeros((n, dim_feature))
        # features_flip = np.zeros_like(features)
        # img_batches = torch.zeros((n, self.channel, h, w))
        idn_batches = torch.zeros((n, ))

        ''' Extracting features '''
        batch_idx = 0
        for batch in tqdm(self.eval_loader, desc='Extracting features'):
            img, idn = batch

            img = img.cuda()
            idn = idn.cuda()

            feat, seg = self._infer(img)

            features[batch_idx * self.batch_size:
                     batch_idx * self.batch_size + self.batch_size] = feat.cpu().numpy()
            # img_batches[batch_idx * self.batch_size:
            #             batch_idx * self.batch_size + self.batch_size] = img.cpu()
            idn_batches[batch_idx * self.batch_size:
                        batch_idx * self.batch_size + self.batch_size] = idn.cpu()

            batch_idx += 1

        ''' identification accuracy '''
        print('Extracting finished, start calculating...')
        right, wrong1, wrong2 = 0, 0, 0

        def norm_feats(feats: np.ndarray):
            norm_f = np.sum(feats ** 2, axis=1, keepdims=True)
            return feats / norm_f

        scrub_ori_feats = norm_feats(features[:self.probe_cnt])
        scrub_occ_feats = norm_feats(features[self.probe_cnt: self.probe_cnt * 2])
        gallery_feats = norm_feats(features[self.probe_cnt * 2:])

        dists_ori_gallery = cdist(scrub_ori_feats, gallery_feats, 'cosine')  # (i,j)
        dists_ori_gallery_min = np.min(dists_ori_gallery, axis=1, keepdims=True)  # (i,1)

        dists_ori_occ = cdist(scrub_ori_feats, scrub_occ_feats, 'cosine')  # (i,i)
        dists_ori_occ[np.diag_indices_from(dists_ori_occ)] = 100.  # same images set to very large distance
        dists_ori_occ_min = np.min(dists_ori_occ, axis=1)  # (i,)

        # dists_ori_all = np.concatenate([dists_ori_occ, dists_ori_gallery_min], axis=1)  # (i,i+1)
        # predict_index = dists_ori_all.argmin(axis=1)  # (i,)
        # for idx_ori in range(len(predict_index)):
        #     idx_pred = predict_index[idx_ori]
        #     if idx_pred >= self.probe_cnt:
        #         wrong1 += 1
        #     elif idn_batches[idx_ori] != idn_batches[idx_pred]:
        #         wrong2 += 1
        #     else:
        #         right += 1

        low, high = 0, 0
        while high < dists_ori_occ.shape[0]:
            while high < dists_ori_occ.shape[0] and idn_batches[low] == idn_batches[high]:
                high += 1
            for row in range(low, high):
                for col in range(low, high):
                    if row == col:
                        continue
                    if dists_ori_gallery_min[row] < dists_ori_occ[row][col]:
                        wrong1 += 1
                    else:
                        right += 1
            low = high

        ret_acc = right / (wrong1 + wrong2 + right)
        print('Identification finished, right=%d, wrong1=%d, wrong2=%d, total=%d, acc=%.2f%%' % (
            right, wrong1, wrong2, right + wrong1 + wrong2, ret_acc * 100
        ))

        ''' tar@far '''
        far_val = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        ret_tarfar = np.zeros((len(far_val)))
        return ret_acc, ret_tarfar

    def start_eval(self):
        if self.eval_mode == '1:1':
            ret_acc, ret_tarfar = self._1to1_verification()
        else:
            ret_acc, ret_tarfar = self._1ton_identification()
        return ret_acc, ret_tarfar


def main():
    parser = argparse.ArgumentParser(description='PyTorch MSML Testing')
    parser.add_argument('--network', type=str, default='msml', help='backbone network')
    parser.add_argument('--dataset', type=str, default='lfw',
                        help='lfw, cfp_fp, agedb_30; mfr2, mfv; mega; lfw_tsne, lfw_restore')
    parser.add_argument('--identity_cnt', type=int, default=10, help='used by lfw_tsne')
    parser.add_argument('--weight_folder', type=str, help='the folder containing pre-trained weights')
    parser.add_argument('--protocol', type=str, default='BB', help='add occlusions to the one or two of a pair')
    parser.add_argument('--fill_type', type=str, default='black', help='block occlusion fill type')
    parser.add_argument('--is_vis', type=str, default='no', help='visualization of FM arith')
    parser.add_argument('--no-occ', action='store_true', help='do not add occ')
    args = parser.parse_args()

    args.is_vis = True if args.is_vis == 'yes' else False

    random.seed(4)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    """ Pre-load images into memory """
    print("=> Pre-loading images ...")
    if args.weight_folder is None or (not os.path.exists(args.weight_folder)):
        cfg_path = '/gavin/code/MSML/config.yaml'
    else:
        cfg_path = os.path.join(args.weight_folder, 'config.yaml')
    cfg = load_yaml(cfg_path)
    config_init(cfg)
    if args.network == 'from2021':
        cfg.out_size = (96, 112)  # (w,h)
    print(cfg)

    pre_trans = transforms.Compose([
        transforms.CenterCrop((cfg.out_size[1], cfg.out_size[0])),
    ])
    if cfg.is_gray:
        pre_trans = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(cfg.out_size),
            pre_trans
        ])
    if args.dataset in ('lfw', 'cfp_fp', 'agedb_30'):
        eval_dataset = MXNetEvalDataset(
            dataset_name=args.dataset,
            rec_prefix=cfg.rec,
            norm_0_1=cfg.is_gray,
            pre_trans=pre_trans,
            protocol=args.protocol,
        )
    elif args.dataset in ('mfr2', ):
        pre_trans = transforms.Resize(cfg.out_size)  # MFR2 using Resize is better
        eval_dataset = MFR2EvalDataset(
            norm_0_1=cfg.is_gray,
            pre_trans=pre_trans,
        )
    elif args.dataset in ('mfv', ):
        eval_dataset = MFVEvalDataset(
            norm_0_1=cfg.is_gray,
            pre_trans=pre_trans,
        )
    elif args.dataset in ('mega', ):
        eval_dataset = MegaFaceDataset(
            distract_cnt=-1,
            norm_0_1=cfg.is_gray,
            pre_trans=pre_trans,
            )
    elif args.dataset in ('lfw_tsne', ):
        assert args.is_vis, 'tsne should set args.is_vis as True.'
        eval_dataset = MXNetTsneDataset(
            dataset_name=args.dataset,
            rec_prefix=cfg.rec,
            identity_cnt=args.identity_cnt,
            norm_0_1=cfg.is_gray,
            pre_trans=pre_trans,
        )
    elif args.dataset in ('lfw_restore', ):
        eval_dataset = MXNetRestorerDataset(
            dataset_name=args.dataset,
            rec_prefix=cfg.rec,
            norm_0_1=cfg.is_gray,
            pre_trans=pre_trans,
        )
    else:
        pre_trans = transforms.Compose([])
        eval_dataset = None

    """ Multi-Test """
    # lo_list = [40,]
    # hi_list = [41,]
    lo_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90] if not args.is_vis else [35, ]
    hi_list = [1, 11, 21, 31, 41, 51, 61, 71, 81, 91] if not args.is_vis else [36, ]
    if args.no_occ or args.dataset in ('mfr2', 'mfv', 'mega'):
        lo_list, hi_list = [0], [1]
    assert len(lo_list) == len(hi_list)

    avg_acc_list = []
    fars = np.zeros((len(lo_list), 5))

    for ind in range(0, len(lo_list)):
        print('================== [ %d ] ===============' % ind)

        lo, hi = lo_list[ind], hi_list[ind]
        print('random block range: [%d ~ %d)' % (lo, hi))

        occ_trans = RandomBlock(lo, hi, fill=args.fill_type)
        # occ_trans = RandomConnectedPolygon(
        #     is_training=False
        # )
        # occ_trans = RandomRealObject(
        #     object_path='./datasets/augment/occluder/object_test',
        #     is_training=False
        # )
        if args.dataset in ('mfr2', 'mfv', ):
            occ_trans = transforms.Compose([])
        elif args.dataset in ('mega', ):
            occ_trans = RandomRealObject(
                object_path='./datasets/augment/occluder/object_test',
                is_training=False
            )

        if args.no_occ:
            occ_trans = transforms.Compose([])

        avg_acc = 0.
        repeat_time = 1 if (lo == 0 and hi == 1) or (lo == 100 and hi == 101) or \
                            args.is_vis or args.dataset in ('mfr2', 'mfv', 'mega', ) else 10
        for repeat in range(repeat_time):
            evaluator = MXNetEvaluator(eval_dataset, occ_trans, cfg, args)
            acc, far = evaluator.start_eval()

            avg_acc += acc
            fars[ind] += far

        avg_acc = avg_acc / repeat_time
        fars[ind] /= repeat_time

        avg_acc_list.append(avg_acc)
        print('[avg_acc]: %.4f' % (avg_acc))

    ''' print results '''
    print(cfg)
    print('[target]:', args.dataset, '[protocol]:', args.protocol, '[fill_type]', args.fill_type)
    print('[model_name]:', args.network)
    print('[weight_path]:', args.weight_folder)
    for ind in range(0, len(avg_acc_list)):
        print('[%d ~ %d] | [avg_acc]: %.4f'
              % (lo_list[ind], hi_list[ind], avg_acc_list[ind]))
        far = fars[ind]
        print('          | [tar@far]: %.4f, %.4f, %.4f, %.4f, %.4f'
              % (far[0], far[1], far[2], far[3], far[4]))


if __name__ == "__main__":
    main()
    # CUDA_VISIBLE_DEVICES=0 python3 eval/qeval_mxnet_workers.py --dataset mega --weight_folder out/arc18_
