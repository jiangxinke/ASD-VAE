import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import random
import pickle as pkl

def prepare_structure(adj, structure_feature, use_cuda):
    # ne-hot
    if structure_feature == 'one-hot':
        diag_fts = torch.eye(adj.shape[0])
    elif structure_feature == 'cosine-position-aware':
        if use_cuda:
            # d2c_code
            diag_fts = preselect_anchor(adj, 'cuda', cosine=True)
        else:
            diag_fts = preselect_anchor(adj, 'cpu', cosine=True)
    elif structure_feature == 'position-aware':
        if use_cuda:
            # cosine-d2c
            diag_fts = preselect_anchor(adj, 'cuda', cosine=False)
        else:
            diag_fts = preselect_anchor(adj, 'cpu', cosine=False)
    return diag_fts


def preselect_anchor(A, layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu', cosine=True):
    num_nodes = A.size(0)
    edges = A._indices().numpy()
    dists = precompute_dist_data(edges, num_nodes, approximate=-1)
    dists = torch.from_numpy(dists).float()

    anchorset_id = get_random_anchorset(num_nodes, c=10)
    dists_max, dists_argmax = get_dist_max(anchorset_id, dists, device)
    if cosine:
        cosine_anchor_distance = torch.Tensor(cosine_similarity(dists_max)).float()
        return cosine_anchor_distance
    else:
        return dists_max


def precompute_dist_data(edge_index, num_nodes, approximate=0):
    '''
    Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
    :return:
    '''
    graph = nx.Graph()
    edge_list = edge_index.transpose(1, 0).tolist()
    graph.add_edges_from(edge_list)

    n = num_nodes
    dists_array = np.zeros((n, n))
    dists_dict = nx.all_pairs_shortest_path_length(graph, cutoff=approximate if approximate > 0 else None)
    dists_dict = {c[0]: c[1] for c in dists_dict}
    for i, node_i in enumerate(graph.nodes()):
        shortest_dist = dists_dict[node_i]
        for j, node_j in enumerate(graph.nodes()):
            dist = shortest_dist.get(node_j, -1)
            if dist != -1:
                # dists_array[i, j] = 1 / (dist + 1)
                dists_array[node_i, node_j] = 1 / (dist + 1)
    return dists_array


def all_pairs_shortest_path_length_parallel(graph, cutoff=None):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    results = single_source_shortest_path_length_range(graph, nodes[int(0):int(len(nodes))], cutoff)

    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    return dists_dict


def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict


def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


def split_planetoid_data(dataset_str, num_nodes):
    """

    Parameters
    ----------
    dataset_str: string
        the name of dataset (cora, citeseer)
    num_nodes: int
        the number of nodes

    Returns
    -------
    train_mask: torch.tensor
    val_mask: torch.tensor
    test_mask: torch.tensor

    """
    with open("data/{}/ind.{}.y".format(dataset_str, dataset_str), 'rb') as f:
        y = torch.tensor(pkl.load(f, encoding='latin1'))
    test_idx = parse_index_file("data/{}/ind.{}.test.index".format(dataset_str, dataset_str))
    test_idx = torch.LongTensor(np.array(test_idx))
    train_idx = torch.arange(y.size(0), dtype=torch.long)
    val_idx = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    return train_idx, val_idx, test_idx


def get_random_anchorset(n, c=0.5):
    m = int(np.log2(n))
    copy = int(c*m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n, size=anchor_size, replace=False))
    return anchorset_id


def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0],len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0],len(anchorset_id))).long().to(device)
    for i in range(len(anchorset_id)):
        temp_id = torch.as_tensor(anchorset_id[i], dtype=torch.long)
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:,i] = dist_max_temp
        dist_argmax[:,i] = temp_id[dist_argmax_temp]
    return dist_max, dist_argmax


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
