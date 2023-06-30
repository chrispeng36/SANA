# -*- coding: utf-8 -*-
# @Time : 2023/3/23 21:06
# @Author : ChrisPeng
# @FileName: multi_GATv2.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/

import numpy as np
import torch
import dgl
from dgl.nn.pytorch import SGConv, GraphConv, GATConv, GATv2Conv
import torch.nn as nn
from torch.nn import init


def init_weight(modules, activation):
    '''

    :param modules:
    :param activation:
    :return:
    '''
    for m in modules:
        if isinstance(m, nn.Linear):
            if activation is None:
                m.weight.data = init.xavier_uniform_(m.weight.data)
            else:
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain(activation.lower()))
            if m.bias is not None:
                m.bias.data = init.constant_(m.bias.data, 0.0)


class MultiGATv2(nn.Module):
    def __init__(self, input_dim, output_dim, source_feats, target_feats, num_SGC_blocks):
        super(MultiGATv2, self).__init__()
        self.source_feats = source_feats
        self.target_feats = target_feats
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_SGC_blocks = num_SGC_blocks

        self.SGCs = []
        '''
        GATConv的in_feats表示输入特征的维度
        output_dim表示的是输出特征的维度
        所以第一个GAT的输入应该是feats_len

        '''
        SGCconv1 = GATv2Conv(in_feats=source_feats.shape[1], out_feats=output_dim, num_heads=3, bias=True,
                             allow_zero_in_degree=True, activation=torch.nn.Tanh())
        SGCconv2 = GATv2Conv(in_feats=output_dim * 3, out_feats=output_dim, num_heads=3, bias=True, allow_zero_in_degree=True,
                             activation=torch.nn.Tanh())
        self.SGCs.append(SGCconv1)
        self.SGCs.append(SGCconv2)

        self.SGCs = nn.ModuleList(self.SGCs)
        init_weight(self.modules(), activation='tanh')

    def forward(self, graph, net='s', new_feats=None):
        '''

        :param graph:
        :param net:
        :param new_feats:
        :return:
        '''
        if new_feats is not None:
            input = new_feats
        elif net == 's':  # 源网络
            input = self.source_feats
        else:
            input = self.target_feats
        emb_input = input.clone()
        outputs = [emb_input]
        for i in range(self.num_SGC_blocks):
            SGC_output_i = self.SGCs[i](graph, emb_input)
            SGC_output_i = SGC_output_i.reshape(SGC_output_i.shape[0], SGC_output_i.shape[1] * SGC_output_i.shape[2])
            outputs.append(SGC_output_i)
            emb_input = SGC_output_i
            # print(SGC_output_i.shape)

        return outputs