import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from deeprobust.graph.global_attack import Random
import random
from scipy.sparse import csr_matrix
import torch

def attack_adj(ptb_rate, seed, adj, r_type='flip'):
    # adj = adj.to_dense()
    adj = csr_matrix(adj)
    attacker = Random()
    random.seed(seed)
    n_perturbations = int(ptb_rate * (adj.sum() // 2))
    attacker.attack(adj, n_perturbations, type=r_type)
    perturbed_adj = attacker.modified_adj

    return torch.Tensor(perturbed_adj.data)
