# -*- coding: utf-8 -*-
# @Time : 2023/3/23 14:51
# @Author : ChrisPeng
# @FileName: test.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/

import dgl
import numpy as np
import torch as th
from dgl.nn.pytorch import GATv2Conv

# g = dgl.graph(([0, 1, 2, 3, 2, 5], [1, 2, 3, 4, 0, 3]))
# feat = th.ones(6, 3)
# g = dgl.add_self_loop(g)
# conv1 = GCN2Conv(3, layer=1, alpha=0.5, project_initial_features=True, allow_zero_in_degree=True)
# conv2 = GCN2Conv(3, layer=2, alpha=0.5, project_initial_features=True, allow_zero_in_degree=True)
# conv3 = GCN2Conv(3, layer=3, alpha=0.5, project_initial_features=True, allow_zero_in_degree=True)
#
# res = feat
# print(res)
# res = conv1(g, res, feat)
# print(res)
# res = conv2(g, res, feat)
# print(res)
# res = conv3(g, res, feat)
# print(res)

# g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
# g = dgl.add_self_loop(g)
# feat = th.ones(6, 10)
# gatv2conv = GATv2Conv(10, 2, num_heads=3)
# res = gatv2conv(g, feat)
# print(res)

import numpy as np
from utils.graph_utils import *
from SGC_utils import *
from input.dataset import Dataset

S = np.load('attribute_s.npy')
S1 = np.load('structure_s.npy')
sim = np.load('total_s.npy')
source_path = r'../../../graph_data/Arenas/origin/graphsage'
target_path = r'../../../graph_data/Arenas/permutation_noise_graph/del-0.05/graphsage'
groundtruth_path = r'../../../graph_data/Arenas/permutation_noise_graph/del-0.05/dictionaries/groundtruth'

source_dataset = Dataset(data_dir=source_path)
source_nodes_num = source_dataset.G.number_of_nodes()
target_dataset = Dataset(data_dir=target_path)
full_dict = load_gt(groundtruth_path, source_dataset.id2idx, target_dataset.id2idx, 'dict')
acc, MAP, top5, top10 = get_statistics(sim, full_dict, use_greedy_match=False, get_all_metric=True)
acc0, MAP0, top5_0, top10_0 = get_statistics(S, full_dict, use_greedy_match=False, get_all_metric=True)
acc1, MAP1, top5_1, top10_1 = get_statistics(S1, full_dict, use_greedy_match=False, get_all_metric=True)

print("*" * 20, "三层的结果", "*" * 20)
print("Accuracy: {:.4f}".format(acc))
print("MAP: {:.4f}".format(MAP))
print("Precision_5: {:.4f}".format(top5))
print("Precision_10: {:.4f}".format(top10))

print("*" * 20, "输入属性三层的结果", "*" * 20)
print("Accuracy: {:.4f}".format(acc0))
print("MAP: {:.4f}".format(MAP0))
print("Precision_5: {:.4f}".format(top5_0))
print("Precision_10: {:.4f}".format(top10_0))

print("*" * 20, "输入结构三层的结果", "*" * 20)
print("Accuracy: {:.4f}".format(acc1))
print("MAP: {:.4f}".format(MAP1))
print("Precision_5: {:.4f}".format(top5_1))
print("Precision_10: {:.4f}".format(top10_1))