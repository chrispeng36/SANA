# -*- coding: utf-8 -*-
# @Time : 2023/3/16 10:51
# @Author : ChrisPeng
# @FileName: refine_util.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/

from collections import defaultdict
import scipy.sparse as sp
import numpy as np
from scipy.sparse import csr_matrix

def score_top1(alignment_matrix, groundtruth):
    '''
    计算1-1对齐的评分
    :param alignment_matrix:
    :param groundtruth:
    :return:
    '''
    row = np.arange(len(alignment_matrix))  # row是所有的源网络节点index
    col = [np.argmax(alignment_matrix[i]) for i in range(len(alignment_matrix))]  # 找到最大值对应的target的index
    val = np.ones(len(alignment_matrix))
    result = csr_matrix((val, (row, col)), shape=alignment_matrix.shape)
    n_matched = 0

    for key, value in groundtruth.items():
        if result[key, value] == 1:
            n_matched += 1
    return n_matched / len(groundtruth)

def score_alignment_matrix(alignment_matrix, topk = 1, topk_score_weighted = False, true_alignments = None):
    '''

    :param alignment_matrix: 给定对齐矩阵
	:param topk: 约束为1-1对齐
	:param topk_score_weighted:
	:param true_alignments: groundtruth
	:return:
    '''
    n_nodes = alignment_matrix.shape[0]
    correct_nodes = defaultdict(list)
    alignment_score = defaultdict(int)

    if sp.issparse(alignment_matrix):
        if alignment_matrix.shape[0] > 2e4:
            return score_sparse_alignment_matrix(alignment_matrix, topk, topk_score_weighted, true_alignments)
        else:  # convert to dense if small enough
            alignment_matrix = alignment_matrix.toarray()

    if not sp.issparse(alignment_matrix): # 从小到大进行排序
        sorted_indices = np.argsort(alignment_matrix)

    for node_index in range(n_nodes):
        target_alignment = node_index
        if true_alignments is not None:
            target_alignment = int(true_alignments[node_index])
        if sp.issparse(alignment_matrix):
            row, possible_alignments, possible_values = sp.find(alignment_matrix[node_index])
            node_sorted_indices = possible_alignments[possible_values.argsort()]
        else:
            node_sorted_indices = sorted_indices[node_index]

        node_sorted_indices = node_sorted_indices.T.ravel()

        if type(topk) is int: topk = [topk]
        for kval in topk:
            if target_alignment in node_sorted_indices[-kval:]:
                if topk_score_weighted:
                    alignment_score[kval] += 1.0 / (n_nodes - np.argwhere(sorted_indices[node_index] == target_alignment)[0])
                else:
                    alignment_score[kval] += 1
                correct_nodes[kval].append(node_index)

    for kval in topk: alignment_score[kval] /= float(n_nodes)  # normalize
    if len(topk) == 1: alignment_score = alignment_score[
        topk[0]]  # only wanted one score, so return just that one score

    return alignment_score, correct_nodes

def score_sparse_alignment_matrix(alignment_matrix, topk=1, topk_score_weighted=False, true_alignments=None):
    n_nodes = alignment_matrix.shape[0]
    correct_nodes = defaultdict(list)
    alignment_score = defaultdict(int)

    sparse_format = alignment_matrix.getformat()
    if not sparse_format == "lil":
        alignment_matrix = alignment_matrix.tolil()

    for node_index in range(n_nodes):
        target_alignment = node_index  # default: assume identity mapping, and the node should be aligned to itself
        if true_alignments is not None:  # if we have true alignments (which we require), use those for each node
            target_alignment = int(true_alignments[node_index])

        sorted_indices = np.argsort(alignment_matrix.data[node_index])  # sorted indices nonzero values only
        node_sorted_indices = np.asarray(alignment_matrix.rows[node_index])[
            sorted_indices]  # sorted indices in the whole thing

        if type(topk) is int: topk = [topk]
        for kval in topk:
            if target_alignment in node_sorted_indices[-kval:]:
                if topk_score_weighted:
                    alignment_score[kval] += 1.0 / (
                                n_nodes - np.argwhere(sorted_indices[node_index] == target_alignment)[0])
                else:
                    alignment_score[kval] += 1
                correct_nodes[kval].append(node_index)

    for k in alignment_score: alignment_score[k] /= float(n_nodes)
    if len(topk) == 1: alignment_score = alignment_score[
        topk[0]]  # we only wanted one score: return just this score instead of a dict of scores

    alignment_matrix = alignment_matrix.tocsr()
    return alignment_score, correct_nodes


# Keep only top k entries (topk = None keeps top 1 with ties)
def threshold_alignment_matrix(M, topk=None, keep_dist=False):
    '''

    :param M: 给定的alignment matrix
    :param topk: 规定是top1对齐
    :param keep_dist: 是否保留其原始的距离，否的话置为1
    :return:
    slow, so use dense ops for smaller matrices
    '''
    sparse_input = sp.issparse(M)
    if sparse_input:
        if M.shape[0] > 20000:
            return threshold_alignment_matrix_sparse(M, topk,
                                                     keep_dist)  # big matrix, use sparse format for memory reasons
        else:
            M = M.toarray()  # on smaller matrices, dense is fastest

    if topk is None or topk <= 0:  # top-1, 0-1
        row_maxes = M.max(axis=1).reshape(-1, 1)  # 每一列的最大值
        M[:] = np.where(M == row_maxes, 1, 0)  # keeps ties
        M[M < 1] = 0
        if sparse_input: M = sp.csr_matrix(M)
        return M
    else:  # selects one tie arbitrarily
        ind = np.argpartition(M, -topk)[:, -topk:]  # 记录对齐的结点，1-1
        row_idx = np.arange(len(M)).reshape((len(M), 1)).repeat(topk,
                                                                axis=1)  # n x k matrix of [1...n] repeated k times

        M_thres = np.zeros(M.shape)
        if keep_dist:
            vals = M[row_idx, ind]
            M_thres[row_idx, ind] = vals
        else:
            M_thres[row_idx, ind] = 1
        if sparse_input: M_thres = sp.csr_matrix(M_thres)
        return M_thres