# -*- coding: utf-8 -*-
# @Time : 2023/2/18 12:22
# @Author : ChrisPeng
# @FileName: data_process.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/

import networkx as nx
from networkx.readwrite import json_graph
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class DataPreprocess(object):
    """
    提供俩方法：
    *****  edgelist2networkx: 连边形式的数据集转换为图的形式  *****
    *****  networkx2edgelist: 图形式的数据转换为连边的形式 *****
    """
    @staticmethod
    def edgelist2networkx(edgelist_file,output_dir, features_file=None):
        '''

        :param edgelist_file: 连边信息的数据，1  2表明(1, 2)是图中的一个连边
        :param output_dir: 存储的地方，存储的为G.json和id2idx.json
        :param features_file:
        :return:
        '''
        g = nx.read_edgelist(edgelist_file, nodetype=str)
        # save networkx
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        id2idx = {}
        for idx, node in enumerate(g.nodes()):
            id2idx[node] = idx  # 创建节点->index的映射,按照在边文件中出现的顺序，下标从0开始

        if features_file is not None:
            features_dict = json.load(open(features_file))  # dict:{node_idx:[feature_list]}
            feature_dim = len(list(features_dict.values())[0])  # 属性维度，每个的维度都是一样的
            n_nodes = len(features_dict.keys())  # 节点个数
            features_arr = np.zeros((n_nodes, feature_dim))  # feature_matrix：(n_nodes, feature_dim)
            for id, feature in features_dict.items():
                g.node[id]['feature'] = feature  # 给节点赋予特征
                features_arr[id2idx[id]] = feature  # 给特征矩阵赋值，获取第id个节点的特征
            np.save(os.path.join(output_dir, 'feats.npy'), features_arr)
        with open(os.path.join(output_dir, 'G.json'), 'w+') as file:
            file.write(json.dumps(json_graph.node_link_data(g)))  # 节点连边的关系存储到json文件中
        with open(os.path.join(output_dir, 'id2idx.json'), 'w+') as file:  # id->index的信息写入
            file.write(json.dumps(id2idx))


    @staticmethod
    def network2edgelist(g_file, output_dir):
        g_data = json.load(open(g_file))
        g = json_graph.node_link_graph(g_data)  # 获取节点和连边的关系
        features = None
        # print(g.nodes())#g.nodes()是所有的节点构建的列表
        if 'feature' in g.node[g.nodes()[0]].keys():
            features = {}
            for node in g.nodes():
                features[node] = g.node[node]['feature']
        # save data
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nx.write_edgelist(g, os.path.join(output_dir, 'edgelist'), delimiter=' ', data=False)
        if features is not None:
            with open(os.path.join(output_dir, 'feats.dict'), 'w+') as file:
                file.write(json.dumps(features))

    '''
    下面几个方法是做一些可视化的操作
    '''
    @staticmethod
    def visualize_degree_distribution(G, output_dir, name):
        deg = np.zeros((len(G.nodes()),)).astype(int)
        for i in range(len(deg)):
            deg[i] = len(G.neighbors(G.nodes()[i]))
        sorted_deg = np.sort(deg)

        plt.style.use('seaborn-whitegrid')
        fig = plt.figure()
        axes = fig.gca()
        axes.set_ylim([1, np.max(sorted_deg)])
        axes.plot(range(len(sorted_deg)), sorted_deg, color='blue')
        outfile = output_dir + name + "_deg_dist.png"
        plt.savefig(outfile)
        return deg
        # unique, counts = np.unique(deg, return_counts=True)
        # y_max = max(counts)
        # std = np.std(deg)
        # mean = np.mean(deg)
        # print("Standard deviation: ", std)
        # print("Mean degree: ", mean)
        # print("Max degree: ", max(deg))
        # # quality = round(max(deg) / std, 1)
        # # print("Quality: ", quality)
        # plt.hist(np.array(sorted_deg), int(max(deg) / 2))
        # plt.vlines(mean - std, 0, y_max, linestyle='dashed', linewidth=0.8, label='std line')
        # plt.vlines(mean + std, 0, y_max, linestyle='dashed', linewidth=0.8)
        # plt.vlines(mean, 0, y_max, color='red', linestyle='dashed', linewidth=1, label='mean line')
        # plt.text(mean +std*0.05, y_max*2/3, 'mean = ' + str(round(mean, 1)) + ', std = ' + str(round(std, 1)), color='b')
        # # plt.text(max(deg) / 2, y_max/2, 'quality = ' + str(quality))
        # plt.xlabel('degree')
        # plt.ylabel('num nodes')
        # plt.title('deg')
        # plt.grid(True)
        # outfile = output_dir + name + "_deg.png"
        # plt.savefig(outfile)
        # print("Degree distribution saved to %s" % outfile)

    @staticmethod
    def visualize_distribution(dist, output_dir, name):
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure()
        axes = fig.gca()
        axes.hist(dist, normed=False, bins=30)
        outfile = output_dir + name + "_diff_dist.png"
        plt.savefig(outfile)

    @staticmethod
    def visualize_line_distribution(dist1, dist2, output_dir, name):
        idxs = range(len(dist1))
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure()
        axes = fig.gca()
        axes.set_ylim([1, max(np.max(dist1), np.max(dist2))])
        axes.plot(idxs, dist1, color='red')
        axes.plot(idxs, dist2, color='blue')
        outfile = output_dir + name + "_deg_dist.png"
        plt.savefig(outfile)

    @staticmethod
    def evaluateDataset(dataset1, dataset2, groundtruth, output_dir):
        '''
        统计图的详细信息
        :param dataset1:
        :param dataset2:
        :param groundtruth:
        :param output_dir:
        :return:
        '''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        outfile = open(os.path.join(output_dir, 'statistics.txt'), 'w+')

        print("Source dataset info:")
        outfile.write("Source dataset info:\n")
        print(nx.info(dataset1.G))
        outfile.write(nx.info(dataset1.G))

        print("Target dataset info:")
        outfile.write("\nTarget dataset info:\n")
        print(nx.info(dataset2.G))
        outfile.write(nx.info(dataset2.G))
        target_deg = DataPreprocess.visualize_degree_distribution(dataset2.G, output_dir, "dataset2")
        source_deg = DataPreprocess.visualize_degree_distribution(dataset1.G, output_dir, "dataset1")

        # gt_src_deg = []
        # gt_trg_deg = []
        # scale_ratio = np.mean(source_deg) / np.mean(target_deg)
        # for source_key in groundtruth.keys():
        #     source_id = source_key
        #     target_id = groundtruth[source_id]
        #     source_idx = dataset1.id2idx[source_id]
        #     target_idx = dataset2.id2idx[target_id]
        #     gt_src_deg.append(source_deg[source_idx])
        #     gt_trg_deg.append(int(target_deg[target_idx] * scale_ratio))

        # DataPreprocess.visualize_line_distribution(gt_src_deg, gt_trg_deg, output_dir, "source_target_gt")

        gt_diff_deg = []
        for source_key in groundtruth.keys():
            source_id = source_key
            target_id = groundtruth[source_id]
            source_idx = dataset1.id2idx[source_id]
            target_idx = dataset2.id2idx[target_id]
            gt_diff_deg.append(source_deg[source_idx] - target_deg[target_idx])

        DataPreprocess.visualize_distribution(gt_diff_deg, output_dir, "source_target_gt")

        stats = 0
        total_stats = 0
        # Check if neibours of source dataset in groundtruth are neighbors in target dataset
        for source_node in groundtruth.keys():
            target_node = groundtruth[source_node]
            source_neighbors = dataset1.G.neighbors(source_node)
            target_neighbors = dataset2.G.neighbors(target_node)
            for neighbor in source_neighbors:
                total_stats += 1
                if neighbor in groundtruth.keys():
                    neighbor_in_target = groundtruth[neighbor]
                    if neighbor_in_target in target_neighbors:
                        stats += 1

        print("Ratio of neighbors in source are also neighbors in target: %.4f" % (stats / total_stats))
        outfile.write("\nRatio of neighbors in source are also neighbors in target: %.4f" % (stats / total_stats))

        # Check if source dataset in groundtruth has the same feature with target dataset
        if (dataset1.features is not None) and (dataset2.features is not None):
            stats = 0
            total_stats = 0
            for source_key in groundtruth.keys():
                total_stats += 1
                source_id = source_key
                target_id = groundtruth[source_id]
                source_idx = dataset1.id2idx[source_id]
                target_idx = dataset2.id2idx[target_id]
                if np.array_equal(dataset1.features[source_idx], dataset2.features[target_idx]):
                    stats += 1

            print("Ratio of same feature groundtruth: %.4f" % (stats / total_stats))
            outfile.write("\nRatio of same feature groundtruth: %.4f" % (stats / total_stats))

        print("Stats has been stored in %s" % (output_dir + 'statistics.txt'))
        outfile.close()
