# -*- coding: utf-8 -*-
# @Time : 2023/2/22 12:58
# @Author : ChrisPeng
# @FileName: get_percession.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/

from evaluation.matcher import top_k, greedy_match
import numpy as np
from scipy.sparse import csr_matrix

def get_nn_alignment_matrix(alignment_matrix, source_personality_mapping, target_personality_mapping, id2idx):
    '''

    :param alignment_matrix:
    :param source_personality_mapping: {id:[persona_list]}
    :param target_personality_mapping:
    :param id2idx:
    :return:
    '''
    # Sparse
    row = np.arange(len(alignment_matrix))
    col = [np.argmax(alignment_matrix[i]) for i in range(len(alignment_matrix))]
    row = [id2idx[source_personality_mapping[idx]] for idx in row]
    col = [id2idx[target_personality_mapping[idx]] for idx in col]
    count = 0
    for index in range(len(row)):
        if row[index] == col[index]:
            count += 1
    print(count)
    val = np.ones(len(alignment_matrix))
    result = csr_matrix((val, (row, col)), shape=alignment_matrix.shape)
    return result

def get_stastics(alignment_matrix, groundtruth, id2idx, source_personality_mapping, target_personality_mapping, use_greedy_match=False, get_all_metric = False):
    '''

    :param alignment_matrix:
    :param groundtruth:
    :param source_personality_mapping:
    :param target_personality_mapping:
    :param use_greedy_match:
    :param get_all_metric:
    :return:
    '''
    if use_greedy_match:
        print("this is greedy match accuracy")
        pred = greedy_match(alignment_matrix)
    else:
        pred = get_nn_alignment_matrix(alignment_matrix, source_personality_mapping, target_personality_mapping,id2idx)
    acc = compute_accuracy(pred, groundtruth)
    return acc

def compute_accuracy(pred, gt):
    n_matched = 0
    if type(gt) == dict:
        for key, value in gt.items():
            if pred[key, value] >= 1:
                n_matched += 1
        print(n_matched, ":", len(gt))
        return n_matched / len(gt)


