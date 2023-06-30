# -*- coding: utf-8 -*-
# @Time : 2023/3/22 18:59
# @Author : ChrisPeng
# @FileName: SGC_utils.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/
from evaluation.metrics import get_statistics
import torch
from scipy.sparse import coo_matrix
import numpy as np
import torch.nn.functional as F

def Laplacian_graph(A):
    '''
    给定一幅图的邻接矩阵，返回拉普拉斯图
    :param A:
    :return:
    '''
    for i in range(len(A)):
        A[i, i] = 1 # A_hat = A + I_n
    A = torch.FloatTensor(A) # 转为tensor
    D_ = torch.diag(torch.sum(A, 0) ** (-0.5))
    A_hat = torch.matmul(torch.matmul(D_, A), D_) # D^(-1/2) * A_hat * D^(-1/2)
    A_hat = A_hat.float()
    indices = torch.nonzero(A_hat).t() # 记录非零元素
    values = A_hat[indices[0], indices[1]]
    A_hat = torch.sparse.FloatTensor(indices, values, A_hat.size()) # tensor：3906
    return A_hat, coo_matrix(A.detach().cpu().numpy())

def get_acc(source_outputs, target_outputs, test_dict=None, alphas=None, just_S=False):
    Sf = np.zeros((len(source_outputs[0]), len(target_outputs[0])))
    list_S_numpy = []
    accs = ""
    for i in range(0, len(source_outputs)):
        S = torch.matmul(F.normalize(source_outputs[i]), F.normalize(target_outputs[i]).t())
        S_numpy = S.detach().cpu().numpy()
        if test_dict is not None:
            if not just_S:
                acc = get_statistics(S_numpy, test_dict)
                accs += "Acc layer {} is: {:.4f}, ".format(i, acc)
        if alphas is not None:
            Sf += alphas[i] * S_numpy
        else:
            Sf += S_numpy
    if test_dict is not None:
        if not just_S:
            acc = get_statistics(Sf, test_dict)
            accs += "Final acc is: {:.4f}".format(acc)
    return accs, Sf

def get_all_S(source_outputs, target_outputs,source_outputs_new, target_outputs_new,test_dict=None, alphas=None, just_S=False):
    Sf = np.zeros((len(source_outputs[0]), len(target_outputs[0])))
    list_S_numpy = []
    accs = ""
    for i in range(0, len(source_outputs)):
        S = torch.matmul(F.normalize(source_outputs[i]), F.normalize(target_outputs[i]).t())
        S_new = torch.matmul(F.normalize(source_outputs_new[i]), F.normalize(target_outputs_new[i]).t())
        S_numpy = S.detach().cpu().numpy()
        S_new_numpy = S_new.detach().cpu().numpy()
        if test_dict is not None:
            if not just_S:
                acc = get_statistics(S_numpy, test_dict)
                accs += "Acc layer {} is: {:.4f}, ".format(i, acc)
        if alphas is not None:
            Sf += (alphas[i] * S_numpy + alphas[i] * 0.1 * S_new_numpy)
        else:
            Sf += (S_numpy + 0.1 * S_new_numpy)
    if test_dict is not None:
        if not just_S:
            acc = get_statistics(Sf, test_dict)
            accs += "Final acc is: {:.4f}".format(acc)
    return accs, Sf
