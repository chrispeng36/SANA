import argparse
import numpy as np
import utils.graph_utils as graph_utils
import copy
from input.dataset import Dataset
from utils.graph_utils import load_gt


path = r'../compare_robust_datas/Arenas/origin/graphsage'
path1 = r'..compare_robust_datas/Arenas/attribute_noise/del-0.20-0.1-noise-0.1-attribute_noise/graphsage'
def parse_args():
    parser = argparse.ArgumentParser(description="Create synthetic feature for graph")
    parser.add_argument('--input_data1', default=r'../compare_robust_datas/new_econ/origin/graphsage')
    parser.add_argument('--input_data2', default=r'../compare_robust_datas/new_econ/attribute_noise/del-0.20-0.1-noise/graphsage')
    parser.add_argument('--feature_dim', default=20, type=int)
    parser.add_argument('--ground_truth', default=r'../compare_robust_datas/new_econ/attribute_noise/del-0.20-0.1-noise/dictionaries/groundtruth')
    return parser.parse_args()




def create_features(data1, data2, dim, ground_truth):
    feature1 = create_feature(data1, dim)
    feature2 = create_feature(data2, dim)
    
    inverted_groundtruth = {v:k for k, v in ground_truth.items()}
    for i in range(len(feature2)):
        try:
            feature2[i] = feature1[inverted_groundtruth[i]]
        except:
            continue

    return feature1, feature2

def add_noise_to_feature2(feature2, pa):
    features1 = copy.deepcopy(feature2)
    list_feat = [features1[0]]
    for i in range(len(features1)):
        notin = 1
        for feat in list_feat:
            if (feat == features1[i]).all():
                notin = 0
        if notin:
            list_feat.append(features1[i])
    change_ids = np.random.randint(len(feature2), size=int(pa*len(feature2)))
    for node in range(len(change_ids)):
        j = np.random.randint(0, len(list_feat))
        features1[change_ids[node]] = list_feat[j]
    return features1


def create_featurex(data, dim):
    deg = data.get_nodes_degrees()
    deg = np.array(deg)
    binn = int(max(deg) / dim)
    feature = np.zeros((len(data.G.nodes()), dim))
    for i in range(len(deg)):
        deg_i = deg[i]
        node_i = data.G.nodes()[i]
        node_i_idx = data.id2idx[node_i]
        feature[node_i_idx, int(deg_i/(binn+ 1))] = 1
    return feature

def create_feature(data, dim):
    shape = (len(data.G.nodes()), int(dim))
    features = np.random.uniform(size=shape)
    for i, feat in enumerate(features):
        mask = np.ones(feat.shape, dtype=bool)
        mask[feat.argmax()] = False
        feat[~mask] = 1
        feat[mask] = 0
    return features


if __name__ == "__main__":
    args = parse_args()
    data1 = Dataset(args.input_data1)
    data2 = Dataset(args.input_data2)
    ground_truth = load_gt(args.ground_truth, data1.id2idx, data2.id2idx, 'dict')
    feature1, feature2 = create_features(data1, data2, args.feature_dim, ground_truth)
    feature2 = add_noise_to_feature2(feature2, pa=0.1)
    print(feature2.shape)
    np.save(args.input_data1 + '/feats.npy', feature1)
    np.save(args.input_data2 + '/feats.npy', feature2)
