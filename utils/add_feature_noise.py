# -*- coding: utf-8 -*-
# @Time : 2023/6/8 16:54
# @Author : ChrisPeng
# @FileName: add_feature_noise.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/

import argparse
import numpy as np
import utils.graph_utils as graph_utils
import copy
from input.dataset import Dataset
from utils.graph_utils import load_gt


def add_noise_to_feature2(feature2, pa):

    print("over")
    features1 = copy.deepcopy(feature2)
    list_feat = [features1[0]]
    for i in range(len(features1)):
        notin = 1
        for feat in list_feat:
            if (feat == features1[i]).all():
                notin = 0
        if notin:
            list_feat.append(features1[i])
    # 随机挑选一些节点
    change_ids = np.random.randint(len(feature2), size=int(pa*len(feature2)))
    for node in range(len(change_ids)):
        features1[change_ids[node]] = np.full(20, 0)
    print("11111")
    return features1

def get_feats2(feats1, gt):
    feats2 = np.ones(feats1.shape)
    inverted_gt = {v: k for k, v in gt.items()}
    for i in range(len(feats2)):
        try:
            feats2[i] = feats1[inverted_gt[i]]
        except:
            continue
    return feats2

if __name__ == '__main__':
    origin_graph_path = '../CENALP_graphs/new_ppi/origin/graphsage'
    save_feats_path = '../CENALP_graphs/new_ppi/attribute_noise/del-0.20-0.5-noise/graphsage'
    gt_path = '../CENALP_graphs/new_ppi/attribute_noise/del-0.20-0.5-noise/dictionaries/groundtruth'

    data1 = Dataset(data_dir=origin_graph_path)
    data2 = Dataset(data_dir=save_feats_path)
    ground_truth = load_gt(gt_path, data1.id2idx, data2.id2idx, 'dict')

    feats1 = data1.features
    feats2 = get_feats2(feats1, ground_truth)
    feature2 = add_noise_to_feature2(feats2, pa=0.5)
    np.save(save_feats_path + '/feats.npy', feature2)

