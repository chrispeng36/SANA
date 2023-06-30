# -*- coding: utf-8 -*-
# @Time : 2023/2/18 12:22
# @Author : ChrisPeng
# @FileName: dataset.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/

import json
import os
import argparse
from scipy.io import loadmat
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
# from .data_process import DataPreprocess
import utils.graph_utils as graph_utils

class Dataset:
    """
        this class receives input from graphsage format with predefined folder structure, the data folder must contains these files:
        G.json, id2idx.json, features.npy (optional)

        Arguments:
        - data_dir: Data directory which contains files mentioned above.
        """

    def __init__(self, data_dir, source_nodes=None, add_nodes=False):
        self.data_dir = data_dir
        self._load_id2idx()
        self._load_G()
        self._load_features()
        if add_nodes:
            feature_dim = self.features.shape[1]
            self.add_singleton_nodes(source_nodes, feature_dim)

        # graph_utils.construct_adjacency(self.G, self.id2idx, sparse=False, file_path=self.data_dir + "/edges.edgelist")
        # self.load_edge_features()
        print("Dataset info:")
        print("- Nodes: ", len(self.G.nodes()))
        print("- Edges: ", len(self.G.edges()))

    def _load_G(self):
        '''
        用节点及其连边的信息构建一幅图
        注意：G.json中存储的是(index, index)是逻辑上的一个下标的连边关系
        所以这里也要用self.idx2id将其转换为节点的id来创建这个图
        所以，这里构造的self.G包含的信息都是其id的信息，与索引无关
        '''

        G_data = json.load(open(os.path.join(self.data_dir, "G.json")))
        # 构建图中的结点的连边
        # G_data['links'][i]['source']表示第i个连边的源节点
        # 原始图中连边的信息是用index相连接的，这里要改成用id相连接
        G_data['links'] = [{'source': self.idx2id[G_data['links'][i]['source']], 'target': self.idx2id[G_data['links'][i]['target']]} for i in range(len(G_data['links']))]
        self.G = json_graph.node_link_graph(G_data)
        for node in list(self.G.nodes()):
            if type(node) == int:
                self.G.remove_node(node)

    def _load_id2idx(self):
        '''
        构建id到index的映射，原始图的信息中已经给出
        '''
        id2idx_file = os.path.join(self.data_dir, 'id2idx.json')
        self.id2idx = json.load(open(id2idx_file))
        self.idx2id = {v: k for k, v in self.id2idx.items()}

    def _load_features(self):
        self.features = None
        feats_path = os.path.join(self.data_dir, 'feats.npy')
        if os.path.isfile(feats_path):
            self.features = np.load(feats_path)
        else:
            self.features = None
        return self.features

    def load_edge_features(self):
        self.edge_features = None
        feats_path = os.path.join(self.data_dir, 'edge_feats.mat')
        if os.path.isfile(feats_path):
            edge_feats = loadmat(feats_path)['edge_feats']
            self.edge_features = np.zeros((len(edge_feats[0]),
                                           len(self.G.nodes()),
                                           len(self.G.nodes())))
            for idx, matrix in enumerate(edge_feats[0]):
                self.edge_features[idx] = matrix.toarray()
        else:
            self.edge_features = None
        return self.edge_features

    def get_adjacency_matrix(self, sparse=False):
        return graph_utils.construct_adjacency(self.G, self.id2idx, sparse=False,
                                               file_path=self.data_dir + "/edges.edgelist")

    def get_nodes_degrees(self):
        return graph_utils.build_degrees(self.G, self.id2idx)

    def get_nodes_clustering(self):
        return graph_utils.build_clustering(self.G, self.id2idx)

    def get_edges(self):
        return graph_utils.get_edges(self.G, self.id2idx)

    def check_id2idx(self):
        # print("Checking format of dataset")
        for i, node in enumerate(self.G.nodes()):
            if (self.id2idx[node] != i):
                print("Failed at node %s" % str(node))
                return False
        # print("Pass")
        return True

    def add_singleton_nodes(self, source_nodes, feature_dim):
        new_node = []
        for i in range(self.G.number_of_nodes(), source_nodes):
            new_node.append(str(i))
            self.id2idx[str(i)] = i
            self.idx2id[i] = str(i)
            self.features = np.row_stack((self.features, np.zeros(feature_dim)))
        self.G.add_nodes_from(new_node)

