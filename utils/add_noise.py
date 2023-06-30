# -*- coding: utf-8 -*-
# @Time : 2023/3/8 0:20
# @Author : ChrisPeng
# @FileName: add_noise.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/
from __future__ import print_function, division
import numpy as np
import random
import json
import argparse
from shutil import copyfile
from input.dataset import Dataset
import networkx as nx
from networkx.readwrite import json_graph
import pdb
import copy
import os

'''
1. 将图的邻接矩阵做转置P * A * P^T
2. 以概率ps随机移除图中的边
3. 不要让图中出现孤立的结点
'''


def create_permutation_matrix(shape):
    L = list(range(shape[1]))
    P = np.zeros(shape)
    for i in range(P.shape[0]):
        a = np.random.choice(L, 1)
        P[i][a] = 1
        L.remove(a)
    return P


def permutation(A, P):
    return np.dot(P, np.dot(A, P.T))


def get_degree(A):
    return np.sum(A, axis=1)


def add_noise_to_graphstructure(A, ps):
    degree = get_degree(A)
    num_edges = sum(degree) / 2  # 总共的边的数目
    threshold = int(num_edges * ps)
    rm_count = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                dice = np.random.rand()
                if dice < ps and degree[i] > 1 and degree[j] > 1 and rm_count < threshold:
                    A[i, j] = 0
                    A[j, i] = 0
                    degree[i] -= 1
                    degree[j] -= 1
                    rm_count += 1
    print("删除了", rm_count, "条边")
    return A


path = r'../CENALP_graphs/Arenas/structure_noise/del-0.10-0.2-noise'

def parse_args():
    parser = argparse.ArgumentParser(description="Create permutation attribute_noise graph.")
    parser.add_argument('--input', default='../CENALP_graphs/Arenas/origin/graphsage', help='Path to load data')
    parser.add_argument('--output', default='../CENALP_graphs/Arenas/structure_noise/del-0.30-0.2-noise', help='Path to save.')
    parser.add_argument('--prefix', default=None, help='Dataset prefix')
    parser.add_argument('--ratio', type=float, default=0.3, help='Probability of remove nodes')
    parser.add_argument('--seed', type=int, default=121, help='Random seed')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    source_path = args.input
    source_graph = Dataset(data_dir=source_path).G
    adj = np.array(nx.adjacency_matrix(source_graph).todense())
    shape = adj.shape
    permutation_matrix = create_permutation_matrix(shape)
    '''注意现在试试不加转置的图'''
    adj1 = adj
    # adj1 = permutation(adj, permutation_matrix)
    '''
    构造新的图的格式是：
    id: str
    links: int -> int
    这里的id和idx都是对于当下的图而言的
    而groundtruth可以根据[node_list] * P得到
    由于这里给定的图的index都是从0开始，可以简单地for in range
    '''
    nodes_list = list(source_graph.nodes())
    for node_idx in range(len(nodes_list)):
        nodes_list[node_idx] = int(nodes_list[node_idx])
    # source_nodes = np.array(nodes_list)
    # target_nodes = np.dot(source_nodes, permutation_matrix)
    # groundtruth = {}
    # for i in range(len(permutation_matrix)):
    #     for j in range(len(permutation_matrix[0])):
    #         if permutation_matrix[i][j] != 0:
    #             groundtruth[j] = i
    groundtruth = {}
    for i in range(len(permutation_matrix)):
        groundtruth[i] = i

    ps = args.ratio
    adj1 = add_noise_to_graphstructure(adj1, ps=ps)
    # 构建新的图
    target_graph = nx.from_numpy_matrix(adj1)
    out_dir = args.output
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    id2idx = {}
    for idx, node in enumerate(target_graph.nodes()):
        id2idx[str(node)] = idx
    if not os.path.exists(out_dir + '/graphsage'):
        os.makedirs(out_dir + "/graphsage")
    if not os.path.exists(out_dir + "/edgelist"):
        os.mkdir(out_dir + '/edgelist')
    if not os.path.exists(out_dir + "/dictionaries"):
        os.mkdir(out_dir + '/dictionaries')
    nx.write_edgelist(target_graph, out_dir + "/edgelist/edgelist", delimiter=' ', data=False)
    with open(out_dir + '/graphsage/G.json', 'w+') as file: # 保证links是int -> int
        graph_dict = json_graph.node_link_data(target_graph)
        for i in range(len(graph_dict["nodes"])):
            graph_dict["nodes"][i]["id"] = str(graph_dict["nodes"][i]["id"])
        for i in range(len(graph_dict["links"])):
            graph_dict["links"][i]["source"] = int(graph_dict["links"][i]["source"])
            graph_dict["links"][i]["target"] = int(graph_dict["links"][i]["target"])
        file.write(json.dumps(graph_dict))
    with open(out_dir + '/graphsage/id2idx.json', 'w+') as file:
        file.write(json.dumps(id2idx))
    print(groundtruth)
    with open(out_dir + '/dictionaries/groundtruth', 'w+') as file:
        for anchors in groundtruth.items():
            file.write("{0} {1}\n".format(anchors[0], anchors[1]))

    features = None
    if features is not None:
        np.save(out_dir + '/graphsage/feats.npy', features)

    print("Graph has been saved to ", out_dir)

# if __name__ == '__main__':
#     print(os.path.exists('../graph_data/Arenas/attribute_noise'))




