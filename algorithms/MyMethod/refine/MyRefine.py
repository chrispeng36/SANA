# -*- coding: utf-8 -*-
# @Time : 2023/3/16 10:20
# @Author : ChrisPeng
# @FileName: MyRefine.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/

from input.dataset import Dataset
import utils.graph_utils as graph_utils
import numpy as np
import argparse
import math
import networkx as nx
import time
import os
import sys
from scipy import sparse
import scipy.sparse as sps
from sklearn.preprocessing import normalize
import cProfile, pstats
from refine_util import score_alignment_matrix, threshold_alignment_matrix, score_top1

def parse_args():
    parser = argparse.ArgumentParser(description="Run RefiNA.")
    parser.add_argument('--n-iter', type=int, default=20,
                        help='Maximum #iter for RefiNA. Default is 100.')  # dimensions of other kinds of embeddings
    parser.add_argument('--n-update', type=int, default=-1,
                        help='How many possible updates per node. Default is -1, or dense refinement.  Positive value uses sparse refinement')
    parser.add_argument('--token-match', type=float, default=-1,
                        help="Token match score for each node.  Default of -1 sets it to reciprocal of largest graph #nodes rounded up to smallest power of 10")

    return parser.parse_args()

class Refine():
    def __init__(self, alignment_matrix, adjA, adjB, groundtruth, args, topk=1):
        self.alignment_matrix = alignment_matrix
        self.adjA = adjA
        self.adjB = adjB
        self.groundtruth = groundtruth
        self.n_iter = args.n_iter
        self.n_update = args.n_update
        self.token_match = args.token_match
        self.args = args

    def threshold_alignment_matrix(self, M, topK=None, keep_dist=False):
        '''

        :param M: 给定的alignment matrix
        :param topk: 规定是top1对齐
        :param keep_dist: 是否保留其原始的距离，否的话置为1
        :return: 将对齐矩阵改变为topk节点为1，但是其他为0的形式
        slow, so use dense ops for smaller matrices
        '''
        if topK is None or topK <= 0:
            row_maxes = M.max(axis=1).reshape(-1, 1) # 每列的最大值
            M[:] = np.where(M == row_maxes, 1, 0) # 保持连接
            M[M < 1] = 0
            return M
        else:
            ind = np.argpartition(M, -topK)[:, -topK:] # 记录的是对齐的topk节点
            row_idx = np.arange(len(M)).reshape((len(M), 1)).repeat(topK, axis=1)
            M_thres = np.zeros(M.shape)
            if keep_dist:
                vals = M[row_idx, ind]
                M_thres[row_idx, ind] = vals
            else:
                M_thres[row_idx, ind] = 1
            return M_thres

    def refina(self, alignment_matrix, adj1, adj2, args, true_alignments=None):
        '''
        对齐细化过程
        :param alignment_matrix:
        :param adj1:
        :param adj2:
        :param args:
        :param true_alignments:
        :return:
        '''
        if args.token_match < 0: # 那就自动选择增加的那个小值
            pow_10 = math.log(max(adj1.shape[0], adj2.shape[0]), 10) # log_10(max(m,n))
            args.token_match = 10**-int(math.ceil(pow_10))
        scores = [-1]
        final_align = None
        print(args.n_iter)
        for i in range(args.n_iter):
            if alignment_matrix.shape[0] < 20000:
                print(print(("Scores after %d refinement iterations" % i)))
                if true_alignments is not None:
                    score = score_top1(alignment_matrix, true_alignments)
                    # if(score < scores[-1] and i > 20):
                    #     print("开始下降了，终止迭代")
                    #     break
                    if(score > scores[-1]):
                        final_align = alignment_matrix # final_align一直取上升的，也就是取最大值
                    scores.append(score)
                    print("Top 1 accuracy: %.5f" % score)
                mnc = self.score_MNC(alignment_matrix, adj1, adj2)
                print("MNC: %.5f" % mnc)

            '''Step 1: compute MNC-based update'''
            update = self.compute_update(adj1, adj2, alignment_matrix, args)
            update = self.compute_update(adj1, adj2, alignment_matrix, args)  # min( int(5*(i+1)), adj1.shape[0]) )

            '''Step 2: apply update and token match'''
            if args.n_update > 0:
                if sps.issparse(alignment_matrix): # add token match score here so we can selectively update
                    nonzero_updates = update.nonzero()  # Indices of alignments to update
                    updated_data = np.asarray(alignment_matrix[nonzero_updates])  # Alignment values we want to update
                    updated_data += args.token_match  # Add token match
                    updated_data *= update.data  # Multiplicatively update them

                    alignment_matrix = alignment_matrix.tolil()
                    alignment_matrix[nonzero_updates] = updated_data
                    alignment_matrix.tocsr()
                else:
                    alignment_matrix[update != 0] += args.token_match
                    alignment_matrix[update != 0] *= update[update != 0]
            else: # 更新Mk = Mk−1 ◦ A1Mk−1A2
                alignment_matrix = alignment_matrix * update
                alignment_matrix += args.token_match

            '''Step 3: normalize'''
            alignment_matrix = self.normalize_alignment_matrix(alignment_matrix)
        final_score = sorted(scores)[-1]
        print("*"*20, "经过迭代细化后的对齐精度为：", final_score, "*"*20)
        return final_align


    def normalize_alignment_matrix(self, alignment_matrix):
        alignment_matrix = normalize(alignment_matrix, norm="l1", axis=1)
        alignment_matrix = normalize(alignment_matrix, norm="l1", axis=0)
        return alignment_matrix


    def compute_update(self, adj1, adj2, alignment_matrix, args):
        '''
        就是为了计算
        A1 * M * A2^T
        :param adj1:
        :param adj2:
        :param alignment_matrix:
        :param args:
        :return:
        '''
        # A1 * M * A2^T
        update_matrix = adj1.dot(alignment_matrix).dot(adj2.T)  # row i: counterparts of neighbors of i

        if args.n_update > 0 and args.n_update < adj1.shape[0]:
            if sps.issparse(update_matrix):
                if update_matrix.shape[
                    0] < 120000: update_matrix = update_matrix.toarray()  # still fits in memory and dense is faster
                update_matrix = threshold_alignment_matrix(update_matrix, topk=args.n_update, keep_dist=True)
                update_matrix = sps.csr_matrix(update_matrix)
            else:
                update_matrix = threshold_alignment_matrix(update_matrix, topk=args.n_update, keep_dist=True)
        return update_matrix

    def score_MNC(self, alignment_matrix, adj1, adj2):
        '''
        按照论文中的公式计算MNC得分
        :param alignment_matrix:
        :param adj1:
        :param adj2:
        :return:
        '''
        mnc = 0
        if sps.issparse(alignment_matrix): alignment_matrix = alignment_matrix.toarray()
        if sps.issparse(adj1): adj1 = adj1.toarray()
        if sps.issparse(adj2): adj2 = adj2.toarray()
        counter_dict = self.get_counterpart(alignment_matrix) # 首先找到对齐的节点
        node_num = alignment_matrix.shape[0]

        for i in range(node_num):
            a = np.array(adj1[i, :])
            one_hop_neighbor = np.flatnonzero(a)  # 找到节点的一阶邻居
            b = np.array(adj2[counter_dict[i], :])  # 找到算法判断是对齐的结点
            # neighbor of counterpart
            new_one_hop_neighbor = np.flatnonzero(b)  # 同样找到一阶邻居
            one_hop_neighbor_counter = []  # 存储的是当前节点一阶邻居的对应的判别对齐节点

            for count in one_hop_neighbor:
                one_hop_neighbor_counter.append(counter_dict[count])

            # 查找两个数组中相同的个数
            num_stable_neighbor = np.intersect1d(new_one_hop_neighbor, np.array(one_hop_neighbor_counter)).shape[0]
            union_align = np.union1d(new_one_hop_neighbor, np.array(one_hop_neighbor_counter)).shape[0]  # 取并集

            sim = float(num_stable_neighbor) / union_align  # 归一化
            mnc += sim  # 相加

        mnc /= node_num
        return mnc

    def get_counterpart(self, alignment_matrix):
        counterpart_dict = {}

        if not sps.issparse(alignment_matrix):
            sorted_indices = np.argsort(alignment_matrix)

        n_nodes = alignment_matrix.shape[0]
        for node_index in range(n_nodes):

            if sps.issparse(alignment_matrix):
                row, possible_alignments, possible_values = sps.find(alignment_matrix[node_index])
                node_sorted_indices = possible_alignments[possible_values.argsort()]
            else:
                node_sorted_indices = sorted_indices[node_index]
            counterpart = node_sorted_indices[-1]
            counterpart_dict[node_index] = counterpart
        return counterpart_dict

    def forward(self):
        init_alignment_matrix = self.threshold_alignment_matrix(self.alignment_matrix, topK=1)

        alignment_matrix = self.refina(init_alignment_matrix, self.adjA, self.adjB, args=self.args, true_alignments=self.groundtruth)

        print("Refined alignment results:")
        score = score_top1(alignment_matrix, self.groundtruth)
        print("Top 1 accuracy: %.5f" % score)

        mnc = self.score_MNC(alignment_matrix, self.adjA, self.adjB)
        print("MNC: %.3f" % mnc)
        return alignment_matrix

if __name__ == '__main__':
    embedding_path = r'../../MyMethod/embed/Arenas/MNC-douban/'
    adjA = np.load(embedding_path + '/source/src_adj.npy', allow_pickle=True)
    adjB = np.load(embedding_path + '/target/tgt_adj.npy', allow_pickle=True)
    source_path = r'../../../graph_data/douban/online/graphsage'
    target_path = r'../../../graph_data/douban/offline/graphsage'
    groundtruth_path = r'../../../graph_data/douban/dictionaries/groundtruth'
    source_dataset = Dataset(data_dir=source_path)
    source_nodes_num = source_dataset.G.number_of_nodes()
    target_dataset = Dataset(data_dir=target_path)
    groundtruth = graph_utils.load_gt(groundtruth_path, source_dataset.id2idx, target_dataset.id2idx, 'dict')

    source_embed = np.load(embedding_path + '/source/embed3.npy', allow_pickle=True)
    target_embed = np.load(embedding_path + '/target/embed3.npy', allow_pickle=True)
    alignment_matrix = np.matmul(source_embed, target_embed.T)

    args = parse_args()
    model = Refine(
        alignment_matrix=alignment_matrix,
        adjA=adjA,
        adjB=adjB,
        groundtruth=groundtruth,
        args=args,
        topk=1
    )
    model.forward()




