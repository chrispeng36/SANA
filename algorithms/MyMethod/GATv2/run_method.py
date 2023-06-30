# -*- coding: utf-8 -*-
# @Time : 2023/3/23 21:22
# @Author : ChrisPeng
# @FileName: run_method.py
# @Software: PyCharm
# @Blog ：https://chrispeng36.github.io/

from embedding import EmbeddingModel
from refine.MyRefine import Refine
from input.dataset import Dataset
import utils.graph_utils as graph_utils
import argparse
import numpy as np
from SGC_utils import *
from extract_feature import StructureFeats

def parse_args():
    parser = argparse.ArgumentParser(description="Run RefiNA.")
    parser.add_argument('--n-iter', type=int, default=20,
                        help='Maximum #iter for RefiNA. Default is 100.')  # dimensions of other kinds of embeddings
    parser.add_argument('--n-update', type=int, default=-1,
                        help='How many possible updates per node. Default is -1, or dense refinement.  Positive value uses sparse refinement')
    parser.add_argument('--token-match', type=float, default=-1,
                        help="Token match score for each node.  Default of -1 sets it to reciprocal of largest graph #nodes rounded up to smallest power of 10")

    return parser.parse_args()

if __name__ == '__main__':
    # np.random.seed(121)
    import os
    # print(os.path.exists('D:\pythonProject\MyAlign\graph_data\douban\online\graphsage'))
    source_path = r'../../../graph_data/douban/online/graphsage'
    target_path = r'../../../graph_data/douban/offline/graphsage'
    groundtruth_path = r'../../../graph_data/douban/dictionaries/groundtruth'
    source_dataset = Dataset(data_dir=source_path)
    source_nodes_num = source_dataset.G.number_of_nodes()
    target_dataset = Dataset(data_dir=target_path)
    source_attr = source_dataset.features
    target_attr = target_dataset.features


    # new_attr_s, new_attr_t = features.get_attributes()
    import time
    before = time.time()
    features = StructureFeats(source_dataset=source_dataset,target_dataset=target_dataset)
    new_attr_s, new_attr_t = features.extract_features()

    align = EmbeddingModel(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        groundtruth=groundtruth_path,
        new_attr_s=new_attr_s,
        new_attr_t=new_attr_t
    )
    alignment_matrix = align.align()

    adjA = source_dataset.get_adjacency_matrix().A
    adjB = target_dataset.get_adjacency_matrix().A
    groundtruth = graph_utils.load_gt(groundtruth_path, source_dataset.id2idx, target_dataset.id2idx, 'dict')
    args = parse_args()
    model = Refine(
        alignment_matrix=alignment_matrix,
        adjA=adjA,
        adjB=adjB,
        groundtruth=groundtruth,
        args=args,
        topk=1
    )
    new_align_matrix = model.forward()
    after = time.time()
    acc, MAP, top5, top10 = get_statistics(new_align_matrix, groundtruth, use_greedy_match=False, get_all_metric=True)

    print("*" * 20, "三层的结果", "*" * 20)
    print("Accuracy: {:.4f}".format(acc))
    print("MAP: {:.4f}".format(MAP))
    print("Precision_5: {:.4f}".format(top5))
    print("Precision_10: {:.4f}".format(top10))
    print(after - before)