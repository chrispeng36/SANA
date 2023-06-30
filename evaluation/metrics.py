# encoding: utf-8
import numpy as np
from scipy.sparse import csr_matrix


def get_nn_alignment_matrix(alignment_matrix):
    # Sparse
    row = np.arange(len(alignment_matrix)) # row是所有的源网络节点index
    col = [np.argmax(alignment_matrix[i]) for i in range(len(alignment_matrix))] # 找到最大值对应的target的index
    val = np.ones(len(alignment_matrix))
    result = csr_matrix((val, (row, col)), shape=alignment_matrix.shape)
    return result


def get_statistics(alignment_matrix, groundtruth, groundtruth_matrix=None, use_greedy_match=False, get_all_metric = False):
    if use_greedy_match:
        print("This is greedy match accuracy")
        pred = greedy_match(alignment_matrix)
    else: # 默认不使用贪心的策略，返回的pred是一个对齐的稀疏矩阵
        pred = get_nn_alignment_matrix(alignment_matrix)
    acc = compute_accuracy(pred, groundtruth)

    if get_all_metric:
        MAP, Hit, AUC  = compute_MAP_Hit_AUC(alignment_matrix, groundtruth)
        pred_top_5 = top_k(alignment_matrix, 5)
        top5 = compute_precision_k(pred_top_5, groundtruth)
        pred_top_10 = top_k(alignment_matrix, 10)
        top10 = compute_precision_k(pred_top_10, groundtruth)
        return acc, MAP, top5, top10
    return acc

def compute_precision_k(top_k_matrix, gt):
    n_matched = 0

    if type(gt) == dict:
        for key, value in gt.items():
            if top_k_matrix[key, value] == 1:
                n_matched += 1
        return n_matched/len(gt)

    gt_candidates = np.argmax(gt, axis = 1)
    for i in range(gt.shape[0]):
        if gt[i][gt_candidates[i]] == 1 and top_k_matrix[i][gt_candidates[i]] == 1:
            n_matched += 1

    n_nodes = (gt==1).sum()
    return n_matched/n_nodes

def compute_accuracy(pred, gt):
    n_matched = 0
    if type(gt) == dict:
        for key, value in gt.items():
            if pred[key, value] == 1:
                n_matched += 1
        return n_matched/len(gt)

    for i in range(pred.shape[0]):
        if pred[i].sum() > 0 and np.array_equal(pred[i], gt[i]):
            n_matched += 1
    n_nodes = (gt==1).sum()
    return n_matched/n_nodes

def compute_MAP_Hit_AUC(alignment_matrix, gt):
    MAP = 0
    AUC = 0
    Hit = 0
    for key, value in gt.items():
        ele_key = alignment_matrix[key].argsort()[::-1]
        for i in range(len(ele_key)):
            if ele_key[i] == value:
                ra = i + 1 # r1
                MAP += 1/ra
                Hit += (alignment_matrix.shape[1] + 1) / alignment_matrix.shape[1]
                AUC += (alignment_matrix.shape[1] - ra) / (alignment_matrix.shape[1] - 1)
                break
    n_nodes = len(gt)
    MAP /= n_nodes
    AUC /= n_nodes
    Hit /= n_nodes
    return MAP, Hit, AUC


def greedy_match(S):
    """
    :param S: Scores matrix, shape MxN where M is the number of source nodes,
        N is the number of target nodes.
    :return: A dict, map from source to list of targets.
    """
    S = S.T
    m, n = S.shape
    x = S.T.flatten()
    min_size = min([m,n])
    used_rows = np.zeros((m))
    used_cols = np.zeros((n))
    max_list = np.zeros((min_size))
    row = np.zeros((min_size))  # target indexes
    col = np.zeros((min_size))  # source indexes

    ix = np.argsort(-x) + 1

    matched = 1
    index = 1
    while(matched <= min_size):
        ipos = ix[index-1]
        jc = int(np.ceil(ipos/m))
        ic = ipos - (jc-1)*m
        if ic == 0: ic = 1
        if (used_rows[ic-1] == 0 and used_cols[jc-1] == 0):
            row[matched-1] = ic - 1
            col[matched-1] = jc - 1
            max_list[matched-1] = x[index-1]
            used_rows[ic-1] = 1
            used_cols[jc-1] = 1
            matched += 1
        index += 1

    result = np.zeros(S.T.shape)
    for i in range(len(row)):
        result[int(col[i]), int(row[i])] = 1
    return result

def top_k(S, k=1):
    """
    S: scores, numpy array of shape (M, N) where M is the number of source nodes,
        N is the number of target nodes
    k: number of predicted elements to return
    """
    top = np.argsort(-S)[:,:k]
    result = np.zeros(S.shape)
    for idx, target_elms in enumerate(top):
        for elm in target_elms:
            result[idx,elm] = 1
    return result
