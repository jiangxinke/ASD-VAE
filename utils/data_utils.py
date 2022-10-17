import pickle as pkl
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.utils import shuffle
from deeprobust.graph.global_attack import Random
import random
from scipy.sparse import csr_matrix


class Data:
    def __init__(self, dataset_str, args):
        if dataset_str in ['cora', 'citeseer']:
            data = load_planetoid_data(dataset_str, args)
        elif dataset_str in ['amaphoto', 'amacomp']:
            data = load_amazon_data(dataset_str, args)
        else:
            raise ValueError("Dataset {0} does not exist".format(dataset_str))
        self.adj = data['adj']
        self.edge_list = data['edge_list']
        self.features = data['features']
        self.labels = data['labels']
        self.G = data['G']
        self.num_features = self.features.size(1)

    def to(self, device):
        """

        Parameters
        ----------
        device: string
            cpu or cuda

        """
        self.adj = (self.adj.to_dense().to(device)).to_sparse()
        self.edge_list = self.edge_list.to(device)
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)


class NodeClsData(Data):
    def __init__(self, dataset_str, args):
        super(NodeClsData, self).__init__(dataset_str, args)
        if dataset_str in ['cora', 'citeseer']:
            if not args.use_lcc:        # not use lcc
                train_mask, val_mask, test_mask = split_planetoid_data(dataset_str, self.labels.size(0), setting=args.profile)
            else:
                train_mask, val_mask, test_mask = set_train_val_test_split(seed=args.seed, data=self.labels, split_idx=None, dataset_name=dataset_str)
        else:  # in ['amaphoto', 'amacomp']
            if not args.use_lcc:  # not use lcc
                train_mask, val_mask, test_mask = split_amazon_data(dataset_str, self.labels.size(0), setting=args.profile)
            else:
                train_mask, val_mask, test_mask = set_train_val_test_split(seed=args.seed, data=self.labels, split_idx=None, dataset_name=dataset_str)
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_classes = int(torch.max(self.labels)) + 1
        self.args = args

    def to(self, device):
        """

        Parameters
        ----------
        device: string
            cpu or cuda

        """
        super().to(device)
        self.train_mask = self.train_mask.to(device)
        self.val_mask = self.val_mask.to(device)
        self.test_mask = self.test_mask.to(device)


class LinkPredData(Data):
    def __init__(self, dataset_str, val_ratio=0.05, test_ratio=0.1, seed=None):
        super(LinkPredData, self).__init__(dataset_str)
        np.random.seed(seed)
        train_edges, val_edges, test_edges = split_edges(self.G, val_ratio, test_ratio)
        adjmat = torch.tensor(nx.to_numpy_matrix(self.G) + np.eye(self.features.size(0))).float()
        negative_edges = torch.stack(torch.where(adjmat == 0))

        # Update edge_list and adj to train edge_list, adj, and adjmat
        edge_list = torch.cat([train_edges, torch.stack([train_edges[1], train_edges[0]])], dim=1)
        self.num_nodes = self.G.number_of_nodes()
        self.edge_list = add_self_loops(edge_list, self.num_nodes)
        self.adj = normalize_adj(self.edge_list)
        self.adjmat = torch.where(self.adj.to_dense() > 0, torch.tensor(1.), torch.tensor(0.))

        neg_idx = np.random.choice(negative_edges.size(1), val_edges.size(1) + test_edges.size(1), replace=False)

        self.val_edges = val_edges
        self.neg_val_edges = negative_edges[:, neg_idx[:val_edges.size(1)]]
        self.test_edges = test_edges
        self.neg_test_edges = negative_edges[:, neg_idx[val_edges.size(1):]]

        # For Link Prediction Training
        N = self.features.size(0)
        E = self.edge_list.size(1)
        self.pos_weight = torch.tensor((N * N - E) / E)
        self.norm = (N * N) / ((N * N - E) * 2)

    def to(self, device):
        """

        Parameters
        ----------
        device: string
            cpu or cuda

        """
        super().to(device)
        self.val_edges = self.val_edges.to(device)
        self.neg_val_edges = self.neg_val_edges.to(device)
        self.test_edges = self.test_edges.to(device)
        self.neg_test_edges = self.neg_test_edges.to(device)
        self.adjmat = self.adjmat.to(device)


def load_planetoid_data(dataset_str, args):
    """

    Parameters
    ----------
    dataset_str: string
        the name of dataset (cora, citeseer)

    Returns
    -------
    data: dict

    """
    names = ['tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        with open("data/planetoid/ind.{}.{}".format(dataset_str, name), 'rb') as f:
            if sys.version_info > (3, 0):
                out = pkl.load(f, encoding='latin1')
            else:
                out = objects.append(pkl.load(f))

            if name == 'graph':
                objects.append(out)
            else:
                out = out.todense() if hasattr(out, 'todense') else out
                objects.append(torch.tensor(out))

    tx, ty, allx, ally, graph = tuple(objects)
    test_idx = parse_index_file("data/planetoid/ind.{}.test.index".format(dataset_str))
    sorted_test_idx = np.sort(test_idx)

    if dataset_str == 'citeseer':
        len_test_idx = max(test_idx) - min(test_idx) + 1
        tx_ext = torch.zeros(len_test_idx, tx.size(1))
        tx_ext[sorted_test_idx - min(test_idx), :] = tx
        ty_ext = torch.zeros(len_test_idx, ty.size(1), dtype=torch.int)
        ty_ext[sorted_test_idx - min(test_idx), :] = ty

        tx, ty = tx_ext, ty_ext

    features = torch.cat([allx, tx], dim=0)
    features[test_idx] = features[sorted_test_idx]

    labels = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    labels[test_idx] = labels[sorted_test_idx]

    G = nx.from_dict_of_lists(graph)
    edge_list = adj_list_from_dict(graph)
    edge_list = add_self_loops(edge_list, features.size(0))

    # if lcc
    if args.use_lcc:
        features, labels, edge_list = keep_only_largest_connected_component(features, labels, edge_list)

    # attack adj
    adj, edge_list = normalize_adj(edge_list, args.attack_rate, args.seed, args.r_type)

    data = {
        'adj': adj,
        'edge_list': edge_list,
        'features': features,
        'labels': labels,
        'G': G,
    }
    return data


def load_amazon_data(dataset_str, args):
    """

    Parameters
    ----------
    dataset_str: string
        the name of dataset (amaphoto, amacomp)

    Returns
    -------
    data: dict

    """
    if dataset_str == 'amacomp':
        loader = dict(np.load(r"/data/qinzd/BA/experiment_try/DSAT_v4/data/amazon/amacomp.npz"))
    else:
        with np.load('data/amazon/' + dataset_str + '.npz', allow_pickle=True) as loader:
            loader = dict(loader)

    feature_mat = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                shape=loader['attr_shape']).todense()
    features = torch.tensor(feature_mat)

    adj_mat = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                            shape=loader['adj_shape']).tocoo()
    edges = [(u, v) for u, v in zip(adj_mat.row.tolist(), adj_mat.col.tolist())]
    G = nx.Graph()
    G.add_nodes_from(list(range(features.size(0))))
    G.add_edges_from(edges)

    edges = torch.tensor([[u, v] for u, v in G.edges()]).t()
    edge_list = torch.cat([edges, torch.stack([edges[1], edges[0]])], dim=1)
    edge_list = add_self_loops(edge_list, loader['adj_shape'][0])

    labels = loader['labels']
    labels = torch.tensor(labels).long()

    # if lcc
    if args.use_lcc:
        features, labels, edge_list = keep_only_largest_connected_component(features, labels, edge_list)

    # attack adj
    adj, edge_list = normalize_adj(edge_list, args.attack_rate, args.seed, args.r_type)

    size = 1            # smaller the size
    features = features[:int(adj.shape[0] / size)]
    labels = labels[:int(adj.shape[0] / size)]

    adj = adj.to_dense()
    adj = adj[:int(adj.shape[0] / size), :int(adj.shape[0] / size)]
    adj = adj.to_sparse_csr()

    data = {
        'adj': adj,
        'edge_list': edge_list,
        'features': features,
        'labels': labels,
        'G': G,
    }
    return data


def split_planetoid_data(dataset_str, num_nodes, setting):
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
    with open("data/planetoid/ind.{}.y".format(dataset_str), 'rb') as f:
        y = torch.tensor(pkl.load(f, encoding='latin1'))
    test_idx = parse_index_file("data/planetoid/ind.{}.test.index".format(dataset_str))  # 140*7
    train_idx = torch.arange(y.size(0), dtype=torch.long)
    val_idx = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    # train_idx = torch.arange(y.size(0)+500, y.size(0) + 1500, dtype=torch.long)
    if setting:     # profile
        train_ratio = 0.5
        valid_ratio = 0.1
        shuffled_nodes = shuffle(np.arange(num_nodes), random_state=1024)
        train_idx = torch.LongTensor(shuffled_nodes[:int(train_ratio * num_nodes)])
        val_idx = torch.LongTensor(
            shuffled_nodes[int(train_ratio * num_nodes):int((train_ratio + valid_ratio) * num_nodes)])
        test_idx = torch.LongTensor(shuffled_nodes[int((train_ratio + valid_ratio) * num_nodes):])

    train_mask = index_to_mask(train_idx, num_nodes)
    val_mask = index_to_mask(val_idx, num_nodes)
    test_mask = index_to_mask(test_idx, num_nodes)
    return train_mask, val_mask, test_mask


def split_amazon_data(dataset_str, num_nodes, setting):
    with np.load('data/amazon/' + dataset_str + '_mask.npz', allow_pickle=True) as masks:
        train_idx, val_idx, test_idx = masks['train_idx'], masks['val_idx'], masks['test_idx']

    train_idx = train_idx[train_idx < int(num_nodes)]
    val_idx = val_idx[val_idx < int(num_nodes)]
    test_idx = test_idx[test_idx < int(num_nodes)]
    if setting:     # profile
        train_ratio = 0.5
        valid_ratio = 0.1
        shuffled_nodes = shuffle(np.arange(num_nodes), random_state=1024)
        train_idx = torch.LongTensor(shuffled_nodes[:int(train_ratio * num_nodes)])
        val_idx = torch.LongTensor(
            shuffled_nodes[int(train_ratio * num_nodes):int((train_ratio + valid_ratio) * num_nodes)])
        test_idx = torch.LongTensor(shuffled_nodes[int((train_ratio + valid_ratio) * num_nodes):])

    train_mask = index_to_mask(train_idx, num_nodes)
    val_mask = index_to_mask(val_idx, num_nodes)
    test_mask = index_to_mask(test_idx, num_nodes)

    return train_mask, val_mask, test_mask


def split_edges(G, val_ratio, test_ratio):
    edges = np.array([[u, v] for u, v in G.edges()])
    np.random.shuffle(edges)
    E = edges.shape[0]
    n_val_edges = int(E * val_ratio)
    n_test_edges = int(E * test_ratio)
    val_edges = torch.LongTensor(edges[:n_val_edges]).t()
    test_edges = torch.LongTensor(edges[n_val_edges: n_val_edges + n_test_edges]).t()
    train_edges = torch.LongTensor(edges[n_val_edges + n_test_edges:]).t()
    return train_edges, val_edges, test_edges


def adj_list_from_dict(graph):
    G = nx.from_dict_of_lists(graph)
    coo_adj = nx.to_scipy_sparse_matrix(G).tocoo()
    indices = torch.from_numpy(np.vstack((coo_adj.row, coo_adj.col)).astype(np.int64))
    return indices


def index_to_mask(index, size):
    mask = torch.zeros((size,), dtype=torch.bool)
    mask[index] = 1
    return mask


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def add_self_loops(edge_list, size):
    i = torch.arange(size, dtype=torch.int64).view(1, -1)
    self_loops = torch.cat((i, i), dim=0)
    edge_list = torch.cat((edge_list, self_loops), dim=1)
    return edge_list


def get_degree(edge_list):
    row = edge_list[0]
    deg = torch.bincount(row)
    return deg


def normalize_adj(edge_list, rate, seed, r_type='flip'):        # adj, add, remove, flip
    if r_type != 'none':
        adj = torch.sparse.FloatTensor(edge_list, torch.ones(edge_list.shape[1]))
        after_attack_adj = attack_adj(rate, seed, adj, r_type='flip')

        edge_list = torch.Tensor(after_attack_adj.todense()).to_sparse().indices()

    deg = get_degree(edge_list)  # N*1
    row, col = edge_list
    deg_inv_sqrt = torch.pow(deg.to(torch.float), -0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    weight = torch.ones(edge_list.size(1))
    v = deg_inv_sqrt[row] * weight * deg_inv_sqrt[col]          # D^{-1/2} A D^{-1/2}
    norm_adj = torch.sparse.FloatTensor(edge_list, v)

    # re:  I - D^{-1/2} A D^{-1/2}  (have added self-loop before)
    # norm_adj = torch.eye(norm_adj.shape[0]).to_sparse() - norm_adj
    return norm_adj, edge_list


def re_normalize_adj(edge_list, rate, seed, r_type='flip'):        # adj, add, remove, flip
    if r_type != 'none':
        adj = torch.sparse.FloatTensor(edge_list, torch.ones(edge_list.shape[1]))
        after_attack_adj = attack_adj(rate, seed, adj, r_type='flip')
        edge_list = torch.Tensor(after_attack_adj.todense()).to_sparse().indices()

    adj = torch.sparse.FloatTensor(edge_list, torch.ones(edge_list.shape[1]))
    ident = torch.eye(adj.shape[0]).to_sparse()
    adj_ = adj + ident
    adj_ = adj_.to_dense()
    edge_list = adj_.to_sparse().indices()
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    laplacian = ident - adj_normalized
    return laplacian, edge_list


def attack_adj(ptb_rate, seed, adj, r_type='flip'):
    adj = adj.to_dense()
    adj = torch.where(adj>0, torch.ones_like(adj), torch.zeros_like(adj))
    adj = csr_matrix(adj)
    attacker = Random()
    random.seed(seed)
    n_perturbations = int(ptb_rate * (adj.sum() // 2))
    attacker.attack(adj, n_perturbations, type=r_type)
    perturbed_adj = attacker.modified_adj
    return perturbed_adj


def keep_only_largest_connected_component(dataset_x, dataset_y, dataset_graph):
    lcc = get_largest_connected_component(dataset_x, dataset_y, dataset_graph)          # 用到了x，y， edge_index,
    x_new = dataset_x[lcc]
    y_new = dataset_y[lcc]

    row, col = dataset_graph.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    return x_new, y_new, torch.LongTensor(edges)


def get_largest_connected_component(dataset_x, dataset_y, dataset_graph) -> np.ndarray:
    remaining_nodes = set(range(dataset_x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component(dataset_graph, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))


def get_component(dataset_graph, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = dataset_graph.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
        queued_nodes.update(neighbors)
    return visited_nodes


def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]


def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper


def set_train_val_test_split(seed: int, data, dataset_name: str, split_idx: int = None) -> Data:
    if dataset_name == 'cora':
        dataset_name = 'Cora'
    if dataset_name == 'citeseer':
        dataset_name = 'CiteSeer'
    if dataset_name in [
        "Cora",
        "CiteSeer",
        "PubMed",
        "amaphoto",
        "amacomp",
        "CoauthorCS",
        "CoauthorPhysics",
    ]:
        # Use split from "Diffusion Improves Graph Learning" paper, which selects 20 nodes for each class to be in the training set
        num_val = 5000 if dataset_name == "CoauthorCS" else 1500
        train_mask, val_mask, test_mask = set_per_class_train_val_test_split(
            seed=seed, data=data, num_val=num_val, num_train_per_class=20, split_idx=split_idx,
        )
    else:
        raise ValueError(f"We don't know how to split the data for {dataset_name}")

    return train_mask, val_mask, test_mask


def set_per_class_train_val_test_split(
    seed: int, data, num_val: int = 1500, num_train_per_class: int = 20, split_idx: int = None,
) -> Data:

    if split_idx is None:
        development_seed = 1684992425
        rnd_state = np.random.RandomState(development_seed)
        num_nodes = data.shape[0]
        development_idx = rnd_state.choice(num_nodes, num_val, replace=False)
        test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]

        train_idx = []
        rnd_state = np.random.RandomState(seed)
        for c in range(data.max() + 1):
            class_idx = development_idx[np.where(data[development_idx].cpu() == c)[0]]
            train_idx.extend(rnd_state.choice(class_idx, num_train_per_class, replace=False))

        val_idx = [i for i in development_idx if i not in train_idx]

        train_mask = get_mask(train_idx, num_nodes)
        val_mask = get_mask(val_idx, num_nodes)
        test_mask = get_mask(test_idx, num_nodes)

    else:
        train_mask = split_idx["train"]
        val_mask = split_idx["valid"]
        test_mask = split_idx["test"]

    return train_mask, val_mask, test_mask


def get_mask(idx, num_nodes):
    """
    Given a tensor of ids and a number of nodes, return a boolean mask of size num_nodes which is set to True at indices
    in `idx`, and to False for other indices.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask
