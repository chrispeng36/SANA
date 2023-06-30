# -*- coding: utf-8 -*-
# @Time : 2023/4/2 15:49
# @Author : ChrisPeng
# @FileName: extract_feature.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/
import os
import networkx as nx
import numpy as np
from input.dataset import Dataset
from networkx.readwrite import json_graph
import math


class StructureFeats():
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def merge_graphs(self):
        G1 = self.source_dataset.G  # 源网络的network形式的数据，所有数据都是跟id有关，与index无关
        G2 = self.target_dataset.G
        res1 = json_graph.node_link_data(G1)
        res2 = json_graph.node_link_data(G2)
        # source graph：douban_offline的id2idx的映射
        # target graph：douban_online的id2idx的映射
        id2idx1 = self.source_dataset.id2idx
        id2idx2 = self.target_dataset.id2idx

        '''
                拼接的方法就是把节点直接加进去就好了
                可以这样理解：G1的邻接矩阵是A1，G2的邻接矩阵是A2
                A1和A2按行拼接
                所以有以下的规则：
                source graph：id = str(source_index)
                              index = int(source_index + len(source.nodes()))
                target graph: id = str(target_id)
                              index = int(target_index + len(target.nodes()))
                '''
        new_nodes_idxs1 = np.arange(len(G1.nodes()))
        new_nodes_idxs2 = np.arange(len(G1.nodes()), len(G1.nodes()) + len(G2.nodes()))

        new_nodes = []  # 长度就是源网络+目标网络的节点数
        for idx, node in enumerate(res1["nodes"]):
            original_index = id2idx1[node['id']]  # 获取到源网络的id，str类型
            node['id'] = str(original_index)  # 节点的id直接变为跟index一样了
            new_nodes.append(node)
        for idx, node in enumerate(res2['nodes']):
            original_index = id2idx2[node['id']]
            node['id'] = str(int(original_index) + len(G1.nodes()))
            new_nodes.append(node)
        '''经过上面两个循环，新的graph中的index->id的映射已经变为int(index)->str(index)了'''

        new_id2idx = {}  # 构建id->index的映射，新图中的索引
        for node in new_nodes:
            new_id2idx[node['id']] = int(node['id'])

        # TODO 这里的新的idx2id变成了int(index)->str(index)，为什么还要用原始的连边？
        # 不过这里也应该能够用映射关系给找回到原始的节点了
        new_links = []  # 连边的列表，还是存储的是没有进行新图构建的那种id
        for link in res1["links"]:
            new_source_index = link['source']
            new_target_index = link['target']
            new_links.append({
                'source': new_source_index,
                'target': new_target_index
            })
        for link in res2["links"]:
            # print(link)
            new_source_index = str(int(link["source"]) + len(G1.nodes()))
            new_target_index = str(int(link["target"]) + len(G1.nodes()))
            new_links.append({
                'source': new_source_index,
                'target': new_target_index
            })
        new_features = None
        features1 = self.source_dataset.features
        features2 = self.target_dataset.features

        if features1 is not None and features2 is not None:
            if features1.shape[1] != features2.shape[1]:
                print("Can not create new features due to different features shape.")
            new_features = np.zeros((features1.shape[0] + features2.shape[0], features1.shape[1]))
            for i, feat in enumerate(features1):
                new_features[i] = feat
            for i, feat in enumerate(features2):
                new_features[i + len(G1.nodes())] = feat

        new_res = json_graph.node_link_data(G1)
        new_res["nodes"] = new_nodes
        new_res["links"] = new_links
        G = json_graph.node_link_graph(new_res)

        return G, new_id2idx, new_features, new_nodes_idxs1, new_nodes_idxs2

    def get_khop_neighbors(self, num_nodes, adj, max_layer=2):
        kneighbors_dict = {}
        for node in range(num_nodes):
            neighbors = np.nonzero(adj[node])[-1].tolist()  # 邻居节点
            if len(neighbors) == 0:  # disconnected node
                print("Warning: node %d is disconnected" % node)
                kneighbors_dict[node] = {0: set([node]), 1: set()}
            else:
                if type(neighbors[0]) is list:
                    neighbors = neighbors[0]
                kneighbors_dict[node] = {0: set([node]), 1: set(neighbors) - set([node])}  # 0表示节点，1表示节点的不包含自身的邻居
        all_neighbors = {}
        for node in range(num_nodes):
            all_neighbors[node] = set([node])
            all_neighbors[node] = all_neighbors[node].union(kneighbors_dict[node][1])
        current_layer = 2  # need to at least consider neighbors
        while True:
            if max_layer is not None and current_layer > max_layer: break
            reached_max_layer = True  # whether we've reached the graph diameter

            for i in range(num_nodes):  # 遍历K阶邻居
                # All neighbors k-1 hops away
                neighbors_prevhop = kneighbors_dict[i][current_layer - 1]

                khop_neighbors = set()
                # Add neighbors of each k-1 hop neighbors
                for n in neighbors_prevhop:
                    neighbors_of_n = kneighbors_dict[n][1]
                    for neighbor2nd in neighbors_of_n:
                        khop_neighbors.add(neighbor2nd)

                # Correction step: remove already seen nodes (k-hop neighbors reachable at shorter hop distance)
                khop_neighbors = khop_neighbors - all_neighbors[i]

                # Add neighbors at this hop to set of nodes we've already seen
                num_nodes_seen_before = len(all_neighbors[i])
                all_neighbors[i] = all_neighbors[i].union(khop_neighbors)
                num_nodes_seen_after = len(all_neighbors[i])

                # See if we've added any more neighbors
                # If so, we may not have reached the max layer: we have to see if these nodes have neighbors
                if len(khop_neighbors) > 0:
                    reached_max_layer = False

                # add neighbors
                kneighbors_dict[i][current_layer] = khop_neighbors  # k-hop neighbors must be at least k hops away

            if reached_max_layer:
                break  # finished finding neighborhoods (to the depth that we want)
            else:
                current_layer += 1  # move out to next layer

        return kneighbors_dict

    def get_degree_sequence(self, node_degrees, max_degree, kneighbors, num_buckets=2):
        if num_buckets is not None:
            degree_counts = [0] * int(math.log(max_degree, num_buckets) + 1)
        else:
            degree_counts = [0] * (max_degree + 1)
        for kn in kneighbors:
            weight = 1
            degree = node_degrees[kn]
            if num_buckets is not None:
                try:
                    degree_counts[int(math.log(degree, num_buckets))] += weight
                except:
                    print("Node %d has degree %d and will not contribute to feature distribution" % (kn, degree))
            else:
                degree_counts[degree] += weight
        return degree_counts

    def extract_features(self, num_buckets=2):
        G, id2idx, feats, src_idxs, trg_idxs = self.merge_graphs()
        adj = nx.adjacency_matrix(G)
        num_nodes = G.number_of_nodes()
        khop_neighbors = self.get_khop_neighbors(num_nodes, adj)
        node_degrees = np.ravel(np.sum(adj, axis=0).astype(int))
        max_degree = max(node_degrees)
        num_features = int(math.log(max_degree, num_buckets)) + 1
        feature_matrix = np.zeros((num_nodes, num_features))
        for n in range(num_nodes):
            for layer in khop_neighbors[n].keys():  # construct feature matrix one layer at a time
                if len(khop_neighbors[n][layer]) > 0:
                    # degree sequence of node n at layer "layer"
                    deg_seq = self.get_degree_sequence(node_degrees, max_degree, khop_neighbors[n][layer])
                    # add degree info from this degree sequence, weighted depending on layer and discount factor alpha
                    # 每层邻居的权重是不一样的，(alpha)^(layer)的加权系数，当然还得加上自身
                    feature_matrix[n] += [(0.01 ** layer) * x for x in deg_seq]
        source_feats = feature_matrix[src_idxs]
        target_feats = feature_matrix[trg_idxs]
        return source_feats, target_feats


if __name__ == '__main__':
    source_path = r'../../graph_data/douban/online/graphsage'
    target_path = r'../../graph_data/douban/offline/graphsage'
    print(os.path.exists(source_path))
    source_dataset = Dataset(data_dir=source_path)
    target_dataset = Dataset(data_dir=target_path)
    # source_attr = source_dataset.features.tolist()
    # target_attr = target_dataset.features.tolist()
    #
    # features = ExtractConstructionFeatures(
    #     source_graph=source_dataset.G,
    #     target_graph=target_dataset.G,
    #     attr1=source_attr,
    #     attr2=target_attr,
    #     num_attr=30,
    # )
    #
    # new_attr_s, new_attr_t = features.get_attributes()
    # print(new_attr_s.shape)
    # print(new_attr_t.shape)
    struct_feats = StructureFeats(source_dataset=source_dataset, target_dataset=target_dataset)
    source_feats, target_feats = struct_feats.extract_features()
    print(source_feats.shape)
    print(target_feats.shape)
    print(type(source_feats))
    print(type(target_feats))