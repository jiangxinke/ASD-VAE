"""
version 1.0
date 2021/02/04
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture


class GCN(nn.Module):
    def __init__(self, nfeat, nclass, nhid=64, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid, dropout)
        self.gc2 = GCNConv(nhid, nclass, dropout)

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.fc = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        self.fc.bias.data.fill_(0)

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        x = torch.spmm(adj, x)
        return x


class GCNLayer(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, dropout):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                              requires_grad=True)

    def forward(self, input, sp_adj, is_sparse_input=False):
        if is_sparse_input:
            h = torch.spmm(input, self.W)
        else:
            h = torch.mm(input, self.W)
        h_prime = torch.spmm(sp_adj, h)
        return F.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(in_features, out_features).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                              requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(out_features, 1).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                               requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(out_features, 1).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                               requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, is_sparse_input=False):
        if is_sparse_input:
            h = torch.spmm(input, self.W)
        else:
            h = torch.mm(input, self.W)
        N = h.size()[0]
        f_1 = h @ self.a1
        f_2 = h @ self.a2
        e = self.leakyrelu(f_1 + f_2.transpose(0, 1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class NormalGCN(nn.Module):
    def __init__(self, data, nhid=16, dropout=0.5):
        super(NormalGCN, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes

        self.gc1 = GCNConv(nfeat, nhid, dropout)
        self.gc2 = GCNConv(nhid, nclass, dropout)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        x = self.gc1(x, adj)
        self.embedding = x
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class baselineGCN(nn.Module):
    def __init__(self, data, nhid=16, dropout=0.5):
        super(baselineGCN, self).__init__()
        nfeat, nclass = data.num_features, data.num_classes

        self.gc1 = GCNConv(nfeat, nhid, dropout)
        self.gc2 = GCNConv(nhid, nclass, dropout)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, data):
        x, adj = data.features, data.adj
        mask = torch.isnan(x)
        x = x * (mask == False) + 0 * (mask == True)
        x[mask==True] = 0
        x = self.gc1(x, adj)
        self.embedding = x
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)



def ex_relu(mu, sigma):
    is_zero = (sigma == 0)
    sigma[is_zero] = 1e-10
    sqrt_sigma = torch.sqrt(sigma)
    w = torch.div(mu, sqrt_sigma)
    nr_values = sqrt_sigma * (torch.div(torch.exp(torch.div(- w * w, 2)), np.sqrt(2 * np.pi)) +
                              torch.div(w, 2) * (1 + torch.erf(torch.div(w, np.sqrt(2)))))
    nr_values = torch.where(is_zero, F.relu(mu), nr_values)
    return nr_values


def init_gmm(features, n_components):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    init_x = imp.fit_transform(features)
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag').fit(init_x)
    return gmm
