# -*- coding: utf-8 -*-
# @Time : 2023/3/22 18:58
# @Author : ChrisPeng
# @FileName: SGC_embedding.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/

from input.dataset import Dataset
import torch
import numpy as np
import networkx as nx
import random
# from network_alignment_model import NetworkAlignmentModel
from utils.graph_utils import *

# from multi_GAT import MultiGAT
# from multi_GCN2CONV import MultiGCN2CONV
from multi_GATv2 import MultiGATv2
import torch.nn.functional as F
import dgl
from SGC_utils import *
from extract_feature import StructureFeats, ExtractConstructionFeatures


class EmbeddingModel():
    def __init__(self, source_dataset, target_dataset, groundtruth, new_attr_s, new_attr_t, act='tanh',
                 num_SGC_blocks=2, input_dim=125, embedding_dim=50, lr=0.01, model_epoch=20, threshold=0.01,
                 beta=0.8, coe_consistency=0.8, noise_level=0, refinement_epochs=0, log=True):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.new_attr_s = new_attr_s
        self.new_attr_t = new_attr_t
        self.alphas = [1, 1, 1]
        self.full_dict = load_gt(groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')  # groundtruth
        cuda = torch.device("cuda:0")
        self.cuda = cuda
        self.act = act
        self.num_SGC_blocks = num_SGC_blocks
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.model_epoch = model_epoch
        self.log = log
        self.beta = beta
        self.threshold = threshold
        self.coe_consistency = coe_consistency
        self.noise_level = noise_level
        self.refinement_epochs = refinement_epochs
        self.save_path = '../embed/Arenas/'

    def embed(self, source_A_hat, target_A_hat, source_feats, target_feats):

        '''将数据转换为torch'''
        # TODO 接下来还要考虑结构属性作为输入的情形
        # TODO 只有输入是不一样的
        new_source_attr = self.new_attr_s
        new_target_attr = self.new_attr_t
        new_source_attr = torch.FloatTensor(new_source_attr)
        new_target_attr = torch.FloatTensor(new_target_attr)
        if self.cuda:
            new_source_attr = new_source_attr.to(self.cuda)
            new_target_attr = new_target_attr.to(self.cuda)
        new_source_attr = F.normalize(new_source_attr)
        new_target_attr = F.normalize(new_target_attr)

        # TODO 如果没有属性信息的话就不用属性作为输入了
        print(source_feats)
        print(target_feats)

        # multi_gat = MultiSGC(
        #     input_dim=self.input_dim,
        #     output_dim=self.embedding_dim,
        #     source_feats=source_features,
        #     target_feats=target_features,
        #     num_SGC_blocks=self.num_SGC_blocks
        # )
        # multi_gat = MultiGCN2CONV(
        #     input_dim=self.input_dim,
        #     output_dim=self.embedding_dim,
        #     source_feats=source_features,
        #     target_feats=target_features,
        #     num_SGC_blocks=self.num_SGC_blocks
        # )
        multi_gat = MultiGATv2(
            input_dim=self.input_dim,
            output_dim=self.embedding_dim,
            source_feats=source_feats,
            target_feats=target_feats,
            num_SGC_blocks=self.num_SGC_blocks
        )  # in_feats -> 48 * 2 -> 48 * 2

        multi_gat_new = MultiGATv2(
            input_dim=self.input_dim,
            output_dim=self.embedding_dim,
            source_feats=new_source_attr,
            target_feats=new_target_attr,
            num_SGC_blocks=self.num_SGC_blocks
        )

        if self.cuda:
            multi_gat = multi_gat.to(self.cuda)
            multi_gat_new = multi_gat_new.to(self.cuda)
        structural_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, multi_gat.parameters()), lr=self.lr)
        structural_optimizer_new = torch.optim.Adam(filter(lambda p: p.requires_grad, multi_gat_new.parameters()),
                                                    lr=self.lr)

        '''随机产生一些噪声图'''
        source_adj = self.source_dataset.get_adjacency_matrix()
        target_adj = self.target_dataset.get_adjacency_matrix()

        # source_edges = nx.edges(nx.from_numpy_matrix(source_adj))
        # target_edges = nx.edges(nx.from_numpy_matrix(target_adj))
        # source_src = []
        # source_dst = []
        # target_src = []
        # target_dst = []
        # for edge in source_edges:
        #     source_src.append(edge[0])
        #     source_dst.append(edge[1])
        # for edge in target_edges:
        #     target_src.append(edge[0])
        #     target_dst.append(edge[1])
        # self.source_graph = dgl.DGLGraph()
        # self.source_graph.add_edges(source_src, source_dst)
        # self.source_graph = dgl.add_self_loop(self.source_graph)
        # self.target_graph = dgl.DGLGraph()
        # self.target_graph.add_edges(target_src, target_dst)
        # self.target_graph = dgl.add_self_loop(self.target_graph)
        self.source_graph = dgl.from_networkx(nx.from_numpy_matrix(source_adj)).to(self.cuda)
        self.target_graph = dgl.from_networkx(nx.from_numpy_matrix(target_adj)).to(self.cuda)
        self.source_graph = dgl.add_self_loop(self.source_graph)
        self.target_graph = dgl.add_self_loop(self.target_graph)

        new_source_adjs = []
        new_target_adjs = []
        new_source_graphs = []
        new_target_graphs = []
        new_source_hats = []
        new_target_hats = []

        source_new_graph_rm, source_new_adj_rm, source_adj_rm = self.graph_augmentation(self.source_dataset,
                                                                                        'remove_edges')
        source_new_graph_add, source_new_adj_add, source_adj_add = self.graph_augmentation(self.source_dataset,
                                                                                           'add_edges')
        new_source_graphs.append(source_new_graph_rm)
        new_source_graphs.append(source_new_graph_add)
        new_source_hats.append(source_new_adj_rm)
        new_source_hats.append(source_new_adj_add)
        new_source_adjs.append(source_adj_rm)
        new_source_adjs.append(source_adj_add)
        new_source_adjs.append(source_adj)
        new_source_graphs.append(self.source_graph)  # 也添加原始的A_hat
        new_source_hats.append(source_A_hat)
        if self.source_dataset.features is not None:
            # print("=======================")
            new_source_feats = self.graph_augmentation(self.source_dataset, 'change_feats')

        target_new_graph_rm, target_new_adj_rm, target_adj_rm = self.graph_augmentation(self.target_dataset,
                                                                                        'remove_edges')
        target_new_graph_add, target_new_adj_add, target_adj_add = self.graph_augmentation(self.target_dataset,
                                                                                           'add_edges')
        new_target_graphs.append(target_new_graph_rm)
        new_target_graphs.append(target_new_graph_add)
        new_target_hats.append(target_new_adj_rm)
        new_target_hats.append(target_new_adj_add)
        new_target_adjs.append(target_adj_rm)
        new_target_adjs.append(target_adj_add)
        new_target_adjs.append(target_adj)
        new_target_graphs.append(self.target_graph)
        new_target_hats.append(target_A_hat)
        if self.target_dataset.features is not None:
            new_target_feats = self.graph_augmentation(self.target_dataset, 'change_feats')

        '''
        有三个图：减去边、增加边和自己
        for j in range(len(new_source_graphs)):
            遍历所有的图(3个)
            计算loss：aug和原本的
        for i in range(2):选择是源网络还是目标网络
        所以每次迭代输出的前三个是源网络，分别对应于减去边、增加边和自己的loss
        所以每次迭代输出的前三个是目标网络，分别对应于减去边、增加边和自己的loss
        '''
        for epoch in range(self.model_epoch):
            if self.log:
                print("Structure learning epoch: {}".format(epoch))
                for i in range(2):  # 判断是源网络还是目标网络
                    for j in range(len(new_source_graphs)):  # 遍历所有的增强图
                        structural_optimizer.zero_grad()
                        structural_optimizer_new.zero_grad()
                        if i == 0:
                            adj1 = source_adj
                            adj2 = new_source_adjs[j]
                            A_hat = source_A_hat
                            augment_A_hat = new_source_hats[j]
                            origin_graph = self.source_graph
                            augment_graph = new_source_graphs[j]
                            # TODO 维度为什么会不一致？
                            outputs = multi_gat(origin_graph, 's')
                            outputs_new = multi_gat_new(origin_graph, 's')
                            if j < 2:
                                augment_outputs = multi_gat(augment_graph, 's')
                                # TODO 改到这里了
                                augment_outputs_new = multi_gat_new(augment_graph, 's')
                            else:
                                if self.source_dataset.features is not None:
                                    augment_outputs = multi_gat(augment_graph, 's', new_source_feats)
                                else:
                                    augment_outputs = multi_gat(augment_graph, 's')
                                augment_outputs_new = multi_gat_new(augment_graph, 's')
                        else:  # 目标网络的情况
                            adj1 = target_adj
                            adj2 = new_target_adjs[j]
                            A_hat = target_A_hat
                            augment_A_hat = new_target_hats[j]
                            origin_graph = self.target_graph
                            augment_graph = new_target_graphs[j]
                            outputs = multi_gat(origin_graph, 't')
                            outputs_new = multi_gat_new(origin_graph, 't')
                            if j < 2:
                                augment_outputs = multi_gat(augment_graph, 't')
                                augment_outputs_new = multi_gat_new(augment_graph, 't')
                            else:
                                if self.target_dataset.features is not None:
                                    augment_outputs = multi_gat(augment_graph, 't', new_target_feats)
                                else:
                                    augment_outputs = multi_gat(augment_graph, 't')
                                augment_outputs_new = multi_gat_new(augment_graph, 't')

                        # 防止网络层数过大造成的损失，inner loss，解决网络内部的传播
                        consistency_loss = self.linkpred_loss(outputs[-1], A_hat)
                        augment_consistency_loss = self.linkpred_loss(augment_outputs[-1], augment_A_hat)
                        consistency_loss = self.beta * consistency_loss + (1 - self.beta) * augment_consistency_loss
                        diff = torch.abs(outputs[-1] - augment_outputs[-1])
                        noise_adaptivity_loss = (diff[diff < self.threshold] ** 2).sum() / len(outputs)
                        # mnc_loss = self.score_MNC(outputs[-1], augment_outputs[-1], adj1, adj2)
                        # if j == 1:
                        #     print("MNC Loss为：", mnc_loss)
                        loss = self.coe_consistency * consistency_loss + (
                                1 - self.coe_consistency) * noise_adaptivity_loss
                        if self.log:
                            print("Loss: {:.4f}".format(loss.data))
                        loss.backward()
                        structural_optimizer.step()

                        # TODO 训练输入结构特征的网络
                        # 防止网络层数过大造成的损失，inner loss，解决网络内部的传播
                        consistency_loss_new = self.linkpred_loss(outputs_new[-1], A_hat)
                        augment_consistency_loss_new = self.linkpred_loss(augment_outputs_new[-1], augment_A_hat)
                        consistency_loss_new = self.beta * consistency_loss_new + (
                                    1 - self.beta) * augment_consistency_loss_new
                        diff_new = torch.abs(outputs_new[-1] - augment_outputs_new[-1])
                        noise_adaptivity_loss_new = (diff_new[diff_new < self.threshold] ** 2).sum() / len(outputs_new)
                        # mnc_loss = self.score_MNC(outputs[-1], augment_outputs[-1], adj1, adj2)
                        # if j == 1:
                        #     print("MNC Loss为：", mnc_loss)
                        loss_new = self.coe_consistency * consistency_loss_new + (
                                1 - self.coe_consistency) * noise_adaptivity_loss_new
                        if self.log:
                            print("structure Loss: {:.4f}".format(loss_new.data))
                        loss_new.backward()
                        structural_optimizer_new.step()
        multi_gat.eval()
        multi_gat_new.eval()
        torch.save(multi_gat, '../../saved_models/douban_gat.pth')
        torch.save(multi_gat_new, '../../saved_models/douban_gat_new.pth')
        return multi_gat, multi_gat_new

    def score_MNC(self, embeddingA, embeddingB, adj1, adj2):
        '''
        应该使得这个指标尽可能的大
        :param embeddingA: 原始网络的这一层的嵌入
        :param embeddingB: 目标网络的这一层的嵌入
        :param adj1:
        :param adj2:
        :return:
        '''
        mnc = 0
        alignment_matrix = torch.matmul(F.normalize(embeddingA), F.normalize(embeddingB).t())
        # alignment_matrix = alignment_matrix.cpu().detach().numpy()
        counterpart_dict = {}
        sorted_indices = alignment_matrix.sort(1)[1]  # 张量
        num_nodes = alignment_matrix.shape[0]
        for node_index in range(num_nodes):
            node_sorted_indices = sorted_indices[node_index]
            counterpart = node_sorted_indices[-1]
            counterpart_dict[node_index] = counterpart

        for i in range(num_nodes):
            a = np.array(adj1[i, :])
            one_hop_neighbor = np.flatnonzero(a)  # 找到节点的一阶邻居
            b = np.array(adj2[counterpart_dict[i], :])  # 找到算法判断是对齐的结点
            # neighbor of counterpart
            new_one_hop_neighbor = np.flatnonzero(b)  # 同样找到一阶邻居

            one_hop_neighbor_counter = []  # 存储的是当前节点一阶邻居的对应的判别对齐节点

            for count in one_hop_neighbor:
                one_hop_neighbor_counter.append(counterpart_dict[count])

            # 查找两个数组中相同的个数
            num_stable_neighbor = np.intersect1d(new_one_hop_neighbor, np.array(one_hop_neighbor_counter)).shape[0]
            union_align = np.union1d(new_one_hop_neighbor, np.array(one_hop_neighbor_counter)).shape[0]  # 取并集

            sim = float(num_stable_neighbor) / union_align  # 归一化
            mnc += sim  # 相加

        mnc /= num_nodes
        mnc_loss = torch.tensor(1 - mnc)
        if self.cuda:
            mnc_loss.to(self.cuda)
        return mnc_loss

    def linkpred_loss(self, embedding, A):
        '''

        :param embedding:
        :param A:
        :return:
        '''
        pred_adj = torch.matmul(F.normalize(embedding), F.normalize(embedding).t())
        if self.cuda:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]).type_as(pred_adj))), dim=1)
        else:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]))), dim=1)
        # linkpred_losss = (pred_adj - A[index]) ** 2
        linkpred_losss = (pred_adj - A) ** 2
        linkpred_losss = linkpred_losss.sum() / A.shape[1]
        return linkpred_losss

    def graph_augmentation(self, dataset, type_aug='remove_edges'):
        '''

        :param dataset:
        :param type_aug:
        :return:
        '''

        edges = dataset.get_edges()
        adj = dataset.get_adjacency_matrix()

        if type_aug == "remove_edges":
            num_edges = len(edges)
            num_remove = int(len(edges) * self.noise_level)
            index_to_remove = np.random.choice(np.arange(num_edges), num_remove, replace=False)
            edges_to_remove = edges[index_to_remove]
            for i in range(len(edges_to_remove)):
                adj[edges_to_remove[i, 0], edges_to_remove[i, 1]] = 0
                adj[edges_to_remove[i, 1], edges_to_remove[i, 0]] = 0
        elif type_aug == "add_edges":
            num_edges = len(edges)
            num_add = int(len(edges) * self.noise_level)
            count_add = 0
            while count_add < num_add:
                random_index = np.random.randint(0, adj.shape[1], 2)
                if adj[random_index[0], random_index[1]] == 0:
                    adj[random_index[0], random_index[1]] = 1
                    adj[random_index[1], random_index[0]] = 1
                    count_add += 1
        elif type_aug == "change_feats":
            feats = np.copy(dataset.features)
            num_nodes = adj.shape[0]
            num_nodes_change_feats = int(num_nodes * self.noise_level)
            node_to_change_feats = np.random.choice(np.arange(0, adj.shape[0]), num_nodes_change_feats, replace=False)
            for node in node_to_change_feats:
                feat_node = feats[node]
                feat_node[feat_node == 1] = 0
                feat_node[np.random.randint(0, feats.shape[1], 1)[0]] = 1
            feats = torch.FloatTensor(feats)
            if self.cuda:
                feats = feats.to(self.cuda)
            return feats

        # graph_edges = nx.edges(nx.from_numpy_matrix(adj))
        # src = []
        # dst = []
        # for edge in graph_edges:
        #     src.append(edge[0])
        #     dst.append(edge[1])
        # new_graph = dgl.DGLGraph()
        # new_graph.add_edges(src, dst)
        # new_graph = dgl.add_self_loop(new_graph)
        new_graph = dgl.from_networkx(nx.from_numpy_matrix(adj))
        new_graph = dgl.add_self_loop(new_graph)
        # new_graph = dgl.add_self_loop(new_graph)
        if self.cuda:
            new_graph = new_graph.to(self.cuda)

        A_hat, _ = Laplacian_graph(adj)
        if self.cuda:
            A_hat = A_hat.to(self.cuda)
        # groundtruth = {}
        # origin_nodes = np.arange(len(adj))
        # new_nodes = permutation_matrix.dot(origin_nodes)
        # for i in range(len(origin_nodes)):
        #     groundtruth[i] = int(new_nodes[i])
        # print(new_graph.num_edges())
        return new_graph, A_hat, adj

    def generate_permutation(self, shape):
        '''
        返回一个permutation matrix
        :param shape:
        :return:
        '''
        L = list(range(shape[1]))
        P = np.zeros(shape)
        for i in range(P.shape[0]):
            a = np.random.choice(L, 1)
            P[i][a] = 1
            L.remove(a)
        return P

    def get_elements(self):
        '''

        :return: 归一化的A矩阵, D^(-1/2) * A_hat * D^(-1/2)
        '''
        source_A_hat, _ = Laplacian_graph(self.source_dataset.get_adjacency_matrix())
        target_A_hat, _ = Laplacian_graph(self.target_dataset.get_adjacency_matrix())
        if self.cuda:
            source_A_hat = source_A_hat.to(self.cuda)
            target_A_hat = target_A_hat.to(self.cuda)

        source_feats = self.source_dataset.features
        target_feats = self.target_dataset.features

        if source_feats is None:  # 如果没有给定特征的话，就赋值为0
            source_feats = np.zeros((len(self.source_dataset.G.nodes()), 1))
            target_feats = np.zeros((len(self.target_dataset.G.nodes()), 1))

        for i in range(len(source_feats)):
            if source_feats[i].sum() == 0:
                source_feats[i, -1] = 1  # 赋值为1
        for i in range(len(target_feats)):
            if target_feats[i].sum() == 0:
                target_feats[i, -1] = 1
        if source_feats is not None:
            source_feats = torch.FloatTensor(source_feats)
            target_feats = torch.FloatTensor(target_feats)
            if self.cuda:
                source_feats = source_feats.to(self.cuda)
                target_feats = target_feats.to(self.cuda)
        source_feats = F.normalize(source_feats)  # 将特征归一化处理
        target_feats = F.normalize(target_feats)
        return source_A_hat, target_A_hat, source_feats, target_feats

    def align(self):
        source_A_hat, target_A_hat, source_feats, target_feats = self.get_elements()
        multi_embed, multi_embed_sturct = self.embed(source_A_hat, target_A_hat, source_feats, target_feats)
        source_outputs = multi_embed(self.source_graph, 's')
        target_outputs = multi_embed(self.target_graph, 't')

        source_outputs_new = multi_embed_sturct(self.source_graph, 's')
        target_outputs_new = multi_embed_sturct(self.target_graph, 't')
        print(source_outputs[0].shape)
        print(source_outputs[1].shape)
        print(source_outputs[2].shape)
        # S0 = torch.matmul(F.normalize(source_outputs[0]), F.normalize(target_outputs[0]).t())
        # S1 = torch.matmul(F.normalize(source_outputs[1]), F.normalize(target_outputs[1]).t())
        # S2 = torch.matmul(F.normalize(source_outputs[2]), F.normalize(target_outputs[2]).t())
        # S0 = S0.detach().cpu().numpy()
        # S1 = S1.detach().cpu().numpy()
        # S2 = S2.detach().cpu().numpy()

        acc, S = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas, just_S=True)
        acc1, S1 = get_acc(source_outputs_new, target_outputs_new, self.full_dict, self.alphas, just_S=True)
        acc_all, sim = get_all_S(
            source_outputs, target_outputs,
            source_outputs_new, target_outputs_new,
            self.full_dict, self.alphas, just_S=True
        )

        np.save("attribute_s.npy", S)
        np.save("structure_s.npy", S1)
        np.save("total_s.npy", sim)
        # print(S)
        # print(S1)

        acc, MAP, top5, top10 = get_statistics(sim, self.full_dict, use_greedy_match=False, get_all_metric=True)
        acc0, MAP0, top5_0, top10_0 = get_statistics(S, self.full_dict, use_greedy_match=False, get_all_metric=True)
        acc1, MAP1, top5_1, top10_1 = get_statistics(S1, self.full_dict, use_greedy_match=False, get_all_metric=True)

        print("*" * 20, "三层的结果", "*" * 20)
        print("Accuracy: {:.4f}".format(acc))
        print("MAP: {:.4f}".format(MAP))
        print("Precision_5: {:.4f}".format(top5))
        print("Precision_10: {:.4f}".format(top10))

        print("*" * 20, "输入属性三层的结果", "*" * 20)
        print("Accuracy: {:.4f}".format(acc0))
        print("MAP: {:.4f}".format(MAP0))
        print("Precision_5: {:.4f}".format(top5_0))
        print("Precision_10: {:.4f}".format(top10_0))

        print("*" * 20, "输入结构三层的结果", "*" * 20)
        print("Accuracy: {:.4f}".format(acc1))
        print("MAP: {:.4f}".format(MAP1))
        print("Precision_5: {:.4f}".format(top5_1))
        print("Precision_10: {:.4f}".format(top10_1))
        #
        # acc0, MAP0, top5_0, top10_0 = get_statistics(S0, self.full_dict, use_greedy_match=False, get_all_metric=True)
        # acc1, MAP1, top5_1, top10_1 = get_statistics(S1, self.full_dict, use_greedy_match=False, get_all_metric=True)
        # acc2, MAP2, top5_2, top10_2 = get_statistics(S2, self.full_dict, use_greedy_match=False, get_all_metric=True)
        #
        # print("*" * 20, "第一层的结果", "*" * 20)
        # print("Accuracy: {:.4f}".format(acc0))
        # print("MAP: {:.4f}".format(MAP0))
        # print("Precision_5: {:.4f}".format(top5_0))
        # print("Precision_10: {:.4f}".format(top10_0))
        #
        # print("*" * 20, "第二层的结果", "*" * 20)
        # print("Accuracy: {:.4f}".format(acc1))
        # print("MAP: {:.4f}".format(MAP1))
        # print("Precision_5: {:.4f}".format(top5_1))
        # print("Precision_10: {:.4f}".format(top10_1))
        #
        # print("*" * 20, "第三层的结果", "*" * 20)
        # print("Accuracy: {:.4f}".format(acc2))
        # print("MAP: {:.4f}".format(MAP2))
        # print("Precision_5: {:.4f}".format(top5_2))
        # print("Precision_10: {:.4f}".format(top10_2))
        return S1


if __name__ == '__main__':
    np.random.seed(121)
    source_path = r'../../../graph_data/fully-synthetic/erdos-renyi-n3000-p5/origin/graphsage'
    target_path = r'../../../graph_data/fully-synthetic/erdos-renyi-n3000-p5/noise/del-0.05/graphsage'
    groundtruth_path = r'../../../graph_data/fully-synthetic/erdos-renyi-n3000-p5/noise/del-0.05/dictionaries/groundtruth'

    source_dataset = Dataset(data_dir=source_path)
    source_nodes_num = source_dataset.G.number_of_nodes()
    target_dataset = Dataset(data_dir=target_path)
    isolates = nx.isolates(source_dataset.G)
    print(list(isolates))
    # source_attr = []
    # target_attr = []
    # features = ExtractConstructionFeatures(
    #     source_graph=source_dataset.G,
    #     target_graph=target_dataset.G,
    #     attr1=source_attr,
    #     attr2=target_attr,
    #     num_attr=30,
    # )
    # new_attr_s, new_attr_t = features.get_attributes()
    features = StructureFeats(source_dataset=source_dataset, target_dataset=target_dataset)
    new_attr_s, new_attr_t = features.extract_features()
    align = EmbeddingModel(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        groundtruth=groundtruth_path,
        new_attr_s=new_attr_s,
        new_attr_t=new_attr_t
    )
    align.align()

'''
10维度的属性
'''