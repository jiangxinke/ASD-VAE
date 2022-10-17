"""
version 1.0
date 2021/02/04
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.sparse as sp


def generate_mask(features, missing_rate, missing_type, setting=False, data=None):
    """

    Parameters
    ----------
    features : torch.tensor
    missing_rate : float
    missing_type : string

    Returns
    -------

    """
    if setting:     # use proflie
        return generate_sat_mask(features, data.train_mask, data.val_mask, data.test_mask)
    else:
        if missing_type == 'uniform':
            return generate_uniform_mask(features, missing_rate)
        if missing_type == 'bias':
            return generate_bias_mask(features, missing_rate)
        if missing_type == 'struct':
            return generate_struct_mask(features, missing_rate)
    raise ValueError("Missing type {0} is not defined".format(missing_type))


def generate_sat_mask(features, mask_1, mask_2, mask_3):
    mask_1 = ~mask_1
    mask = mask_1.repeat([features.shape[1], 1]).T
    return mask


def generate_uniform_mask(features, missing_rate):
    """

    Parameters
    ----------
    features : torch.tensor
    missing_rate : float

    Returns
    -------
    mask : torch.tensor
        mask[i][j] is True if features[i][j] is missing.

    """
    mask = torch.rand(size=features.size())
    mask = mask <= missing_rate
    return mask


def generate_bias_mask(features, ratio, high=0.9, low=0.1):
    """

    Parameters
    ----------
    features: torch.Tensor
    ratio: float
    high: float
    low: float

    Returns
    -------
    mask : torch.tensor
        mask[i][j] is True if features[i][j] is missing.

    """
    node_ratio = (ratio - low) / (high - low)
    feat_mask = torch.rand(size=(1, features.size(1)))
    high, low = torch.tensor(high), torch.tensor(low)
    feat_threshold = torch.where(feat_mask < node_ratio, high, low)
    mask = torch.rand_like(features) < feat_threshold
    return mask


def generate_struct_mask(features, missing_rate):
    """

    Parameters
    ----------
    features : torch.tensor
    missing_rate : float

    Returns
    -------
    mask : torch.tensor
        mask[i][j] is True if features[i][j] is missing.

    """
    node_mask = torch.rand(size=(features.size(0), 1))
    mask = (node_mask <= missing_rate).repeat(1, features.size(1))
    return mask


def apply_mask(features, mask):
    """

    Parameters
    ----------
    features : torch.tensor
    mask : torch.tensor

    """
    features[mask] = float('nan')


def ASD_apply_mask(data, FP, args):
    data.features = data.features * (data.attributes_mask==False) + 0 * (data.attributes_mask==True)
    column_mean = torch.sum(data.features, dim=0)/torch.sum(data.attributes_mask, dim=0)
    # column_mean = torch.mean(data.features, dim=0)
    mean_square = column_mean.repeat(data.features.shape[0]).reshape(data.features.shape[0], data.features.shape[1])

    data.features = data.features * (data.attributes_mask == False) + 0 * (data.attributes_mask == True)

    # data.features = data.features * (data.attributes_mask==False) + mean_square * (data.attributes_mask == True)
    # data.features = data.features * data.attributes_mask + data.features.mean() * (data.attributes_mask == False)
    data.features = data.features * (data.attributes_mask==False) + 0 * (data.attributes_mask == True)
    if FP != 0:
        for i in range(FP):
            data.features = data.adj @ data.features

    data.features = data.ground_truth_feature * (data.attributes_mask == False) + data.features * (
                data.attributes_mask == True)

    data.adj = data.adj.to_dense().to_sparse()

    if args.dataset in ['cora', 'citeseer']:
        # sparse dataset
        # data.adj = (data.adj.to_dense() @ data.adj.to_dense()).to_sparse()      # larger the adjancy
        eps = 1e-6
        if args.filter == 'Katz':
            data.adj = (data.adj.to_dense() @ data.adj.to_dense() @ data.adj.to_dense() @ data.adj.to_dense()).to_sparse()  # larger the adjancy
            beta = 0.05  # rate>=0.7, beta = 0.667
            if args.rate >= 0:
                data.adj = torch.inverse((torch.eye(data.adj.shape[0]) - beta * data.adj.to_dense())) - torch.eye(
                    data.adj.shape[0])
                data.adj = 1 / beta * (data.adj * (data.adj > eps)).to_sparse()
        if args.filter == 'Dense_Katz':
            beta = 0.05
            eps = 1e-3
            if args.rate >= 0:
                data.adj = torch.inverse((torch.eye(data.adj.shape[0]) - beta * data.adj.to_dense())) - torch.eye(
                    data.adj.shape[0])
                data.adj = 1 / beta * (data.adj * (data.adj > eps)).to_sparse()
        if args.filter == 'SGC':
            data.adj = (data.adj.to_dense() @ data.adj.to_dense() @ data.adj.to_dense() @ data.adj.to_dense())  # larger the adjancy
            data.adj = (data.adj * (data.adj > eps)).to_sparse()
        if args.filter == 'RES':
            eps = 1e-9
            beta = 0.7
            data.adj = torch.eye(data.adj.shape[0]) + data.adj
            for i in range(20):
                data.adj = beta * torch.mm(data.adj, data.adj)
            data.adj = (data.adj * (data.adj > eps)).to_sparse()
        if args.filter == 'AGE':
            if args.dataset == 'cora':
                lambda_dataset = 1/1.48
            elif args.dataset == 'citeseer':
                lambda_dataset = 1/1.50
            else:
                lambda_dataset = 1
            # data.adj = torch.eye(data.adj.shape[0]) - lambda_dataset * data.adj.to_dense()      # Lsym
            data.adj = (1-lambda_dataset) * torch.eye(data.adj.shape[0]) + lambda_dataset * data.adj    # Asym
            data.adj = data.adj.to_sparse()

    return data
