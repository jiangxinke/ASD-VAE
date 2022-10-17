from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions.normal import Normal
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Discriminator(nn.Module):
    def __init__(self, n_fts, n_hid, out_fts, dropout):
        super(Discriminator, self).__init__()
        self.dropout = dropout

        self.fc1 = nn.Linear(n_fts, n_hid)
        self.fc2 = nn.Linear(n_hid, out_fts)

    def forward(self, x):
        # make mlp for discriminator
        h1 = self.fc1(x)
        h1 = F.dropout(F.relu(h1), self.dropout, training=self.training)
        h2 = F.sigmoid(self.fc2(h1))
        return h2


class ASD_VAE(nn.Module):
    def __init__(self, n_nodes, n_fts, n_hid, dropout, args, square=True, a_fts=None):
        super(ASD_VAE, self).__init__()
        self.n_fts = n_fts
        self.n_hid = n_hid
        self.dropout = dropout

        self.M = 2

        self.args = args

        # 1. encoder for   A, X
        # 1.1 MLP for X
        self.e_x_1 = nn.Linear(n_fts, 200)  # encoder of e_x 1
        self.e_x_2 = nn.Linear(200, n_hid)  # encoder of e_x 2

        # 1.2 GNN for A(GAE not VGAE)
        if square:
            a_fts_input = n_nodes
        else:
            a_fts_input = a_fts

        if args.enc_name == 'GCN':
            self.GCN1 = GCNLayer(a_fts_input, 200, dropout=dropout).to(device)
            self.GCN2 = GCNLayer(200, n_hid, dropout=dropout).to(device)
        elif args.enc_name == 'GAT':
            self.GCN1 = GATLayer(a_fts_input, 200, dropout=dropout, alpha=args.alpha).to(device)        # fixme
            self.GCN2 = GATLayer(200, n_hid, dropout=dropout, alpha=args.alpha).to(device)
        elif args.enc_name == 'Normal_GCN':
            self.GCN = NormalGCN(a_fts_input, n_hid, dropout=dropout).to(device)

        # 2. encoder for Common features: Ez
        self.e_c_1 = nn.Linear(n_hid, 200)  # encoder for e_c_1
        self.e_c_2 = nn.Linear(200, n_hid * self.M * 2)  # encoder for e_c_2

        # 3. decoders for A, X
        # 3.1 decoder for X
        self.d_x_1 = nn.Linear(n_hid, 200)
        self.d_x_2 = nn.Linear(200, n_fts)

        # 3.2 decoder for A
        self.d_a_1 = nn.Linear(n_hid, n_nodes)
        self.d_a_2 = nn.Linear(n_nodes, n_nodes)

        # 4. learning metric for cosine
        self.linearproj_1 = nn.Linear(n_nodes, n_nodes)
        self.linearproj_2 = nn.Linear(n_nodes, n_nodes)

        # 5. GAN Discriminator
        self.disc = Discriminator(n_hid, n_hid, n_fts, dropout)

        # 6. get p(A), p(X)
        self.linear_x = nn.Linear(n_fts, n_hid)
        self.linear_a = nn.Linear(n_nodes, n_hid)

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        return eps * torch.exp(logvar) + mu

    def pre_encode(self, X, A, A_fts):
        # 1. encode X
        X = F.dropout(X, self.dropout, training=self.training)
        ae_h1 = F.relu(self.e_x_1(X))
        ae_h1 = F.dropout(ae_h1, self.dropout, training=self.training)
        self.z_x_star = self.e_x_2(ae_h1)

        # 2.encode A
        if self.args.enc_name == 'Normal_GCN':
            self.z_a_star = self.GCN(A_fts, A).to(device)
        else:
            gae_h1 = self.GCN1(A_fts, A, is_sparse_input=False).to(device)
            gae_h1 = F.dropout(gae_h1, self.dropout, training=self.training)
            torch.cuda.empty_cache()
            self.z_a_star = self.GCN2(gae_h1, A)

        torch.cuda.empty_cache()

    def product_couple_vae(self, coefficient=3):
        # 1. product them
        self.Z_star = coefficient * self.z_x_star * self.z_a_star

        # 2. Common encoder
        Z_star = F.dropout(self.Z_star, self.dropout, training=self.training)
        Z_star = F.relu(self.e_c_1(Z_star))
        Z_star = F.dropout(Z_star, self.dropout, training=self.training)
        self.z_x_a = self.e_c_2(Z_star)

        # 3. separate them
        self.z_x_mu = self.z_x_a[:, :int(self.z_x_a.shape[1] / 4)]
        self.z_x_logvar = self.z_x_a[:, int(self.z_x_a.shape[1] / 4):int(self.z_x_a.shape[1] / 2)]
        self.z_a_mu = self.z_x_a[:, int(self.z_x_a.shape[1] / 2):int(3 * self.z_x_a.shape[1] / 4)]
        self.z_a_logvar = self.z_x_a[:, int(3 * self.z_x_a.shape[1] / 4):]

        # 4. reparameterize
        self.z_x = self.reparameterize(self.z_x_mu, self.z_x_logvar)
        self.z_a = self.reparameterize(self.z_a_mu, self.z_a_logvar)

    def decode(self):
        # 1. decode features
        fts1 = F.relu(self.d_x_1(self.z_x))
        fts1 = F.dropout(fts1, self.dropout, training=self.training)
        self.x_pre = self.d_x_2(fts1)
        self.x_pre = F.sigmoid(self.x_pre)

        # 2. decode adjancy
        adj_z1 = F.relu(self.d_a_1(self.z_a))
        adj_z1 = F.dropout(adj_z1, self.dropout, training=self.training)
        adj_z1 = self.d_a_2(adj_z1)
        self.a_pre = F.sigmoid(adj_z1)


    def sim(self, a, b, flag):
        if flag == 'x':
            return self.linearproj_1(torch.cosine_similarity(a, b)).unsqueeze(1).repeat(1, a.shape[1])
        else:
            return self.linearproj_2(torch.cosine_similarity(a, b)).unsqueeze(1).repeat(1, a.shape[1])

    def key_probablity(self, sample_prob=0.2):
        # 1. compute q(z_j|v_j) = q(z_j^*|v_j) * sim
        self.q_z_x_v_x = self.z_x_star * self.sim(self.z_x_star, self.z_x, flag='x')
        self.q_z_a_v_a = self.z_a_star * self.sim(self.z_a_star, self.z_a, flag='a')

        # 2. compute \prod q(z_j|v_j) = q(Z|V)
        self.q_Z_V = self.q_z_x_v_x * self.q_z_a_v_a

        # 4. compute \prod q(z_j)
        self.q_z_xa = self.z_x * self.z_a

        # 5. get p(z_j)
        self.p_z_x = Normal(torch.zeros_like(self.q_z_xa), torch.ones_like(self.q_z_xa)).sample()
        self.p_z_a = Normal(torch.zeros_like(self.q_z_xa), torch.ones_like(self.q_z_xa)).sample()

        # 6. p(Z)
        self.p_Z= Normal(torch.zeros_like(self.q_z_xa), 1/2 * torch.ones_like(self.q_z_xa)).sample()

    def loss(self, X, A, idx, weight_mask=None, missing_mask=None, alpha_1=100, alpha_2=11, beta_1=1, beta_2=200,
             beta_3=7, a_pre_true=None):
        # 0. some key functions1
        loss_function = nn.MSELoss(reduction='none')

        EPSILON = 1e-8

        def loss_y(y_true, y_pred, weight_mask=None, missing_mask=None):
            if weight_mask == None:
                # for adjancy
                return torch.mean(loss_function(y_true, y_pred))
            else:
                return torch.mean(loss_function(y_true, y_pred) * weight_mask * missing_mask)

        def log(x):
            return torch.log(x + EPSILON)

        def kld(x, y, idx):
            x = torch.sigmoid(x)
            y = torch.sigmoid(y)
            x = x + EPSILON
            y = y + EPSILON
            return abs((x * log(x / y))[idx].mean())

        self.q_Z = self.q_z_x_v_x * self.linear_x(X) + self.q_z_a_v_a * self.linear_a(A)

        if not self.args.profile:
            idx = torch.arange(0, A.shape[0]).to(device)

        X = X[idx, :]
        x_pre = self.x_pre[idx, :]
        weight_mask = weight_mask[idx, :]
        missing_mask = missing_mask[idx, :]

        # fixme 08.17
        idx = torch.arange(0, A.shape[0]).to(device)
        A = (A.to_dense())[idx, :]
        A = A[:, idx]
        if a_pre_true!=None:
            a_pre = (a_pre_true[:, idx])[idx, :]
        else:
            a_pre = (self.a_pre[:, idx])[idx, :]

        total_loss = 0

        # 1. compute reconstruction loss
        total_loss += alpha_1 * loss_y(X, x_pre, weight_mask, missing_mask)
        total_loss += alpha_2 * loss_y(A, a_pre)

        torch.cuda.empty_cache()

        # 2. compute decoupling loss
        # 2.1 compute \prod q(z_j|v_j) || q(Z)
        Z_star_Z_loss = beta_1 * kld(self.q_Z_V, self.q_Z, idx)
        total_loss += Z_star_Z_loss
        # 2.2 compute q(Z) || \prod(q_z_j)
        decouple_loss = beta_2 * kld(self.q_Z, self.q_z_xa, idx)
        total_loss += decouple_loss
        # 2.3 compute q(z_j) || p(z_j)
        gs_loss = beta_3 * (kld(self.z_x, self.p_z_x, idx) + kld(self.z_a, self.p_z_a, idx))
        total_loss += gs_loss

        # 2.4 compute q(z) || p(z)
        if self.args.common_z:
            total_loss += 1 * kld(self.q_Z, self.p_Z, idx)

        return total_loss

    def forward(self, X, A, diag_fts):
        # 1. pre_encode
        """
        self.z_x_star:  q(z^*_x|v_x)
        self.z_a_star:  q(z^*_a|v_a)
        """
        torch.cuda.empty_cache()
        self.pre_encode(X, A, diag_fts)

        # 2. product and couple
        """
        self.Z_star: q(Z*|V)    
        self.z_x: q(z_x)    self.z_x_mu: mu,  self.z_x_logvar: sigma^2
        self.z_a: q(z_a)    self.z_a_mu: mu,  self.z_a_logvar: sigma^2

        """
        self.product_couple_vae()

        # 3. decode
        """
        self.x_pre: q(v_x)
        self.a_pre: q(v_a)
        """
        torch.cuda.empty_cache()
        self.decode()

        # 4. get key prob
        self.key_probablity()

        return self.x_pre, self.a_pre


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
        torch.cuda.empty_cache()
        e = self.leakyrelu(f_1 + f_2.transpose(0, 1))
        adj = adj.to_dense()
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        return F.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class NormalGCN(nn.Module):
    def __init__(self, in_features, nclass, nhid=200, dropout=0.5):
        super(NormalGCN, self).__init__()
        nfeat = in_features
        self.gc1 = GCNConv(nfeat, nhid, dropout)
        self.gc2 = GCNConv(nhid, nclass, dropout)
        self.dropout = dropout

    def reset_parameters(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return F.elu(x)


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

