import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
from models import NormalGCN, ASD_VAE
from train import NodeClsTrainer, DSAT_NodeClsTrainer
from utils import NodeClsData, apply_mask, generate_mask, prepare_structure, DSAT_apply_mask, attack_adj
import random
import numpy as np
import warnings

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    default='cora',
                    choices=['cora', 'citeseer', 'amacomp', 'amaphoto'],
                    help='dataset name')
parser.add_argument('--type',
                    default='uniform',
                    choices=['uniform', 'bias', 'struct'],
                    help="uniform randomly missing, biased randomly missing, and structurally missing")
parser.add_argument('--rate', default=0.1, type=float, help='missing rate')
parser.add_argument('--attack_rate', default=0.1, type=float, help='attack rate in structre data')
parser.add_argument('--nhid', default=64, type=int, help='the number of hidden units')
parser.add_argument('--dropout', default=0.4, type=float, help='dropout rate')
parser.add_argument('--ncomp', default=5, type=int, help='the number of Gaussian components')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')
parser.add_argument('--epoch', default=2000, type=int, help='the number of training epoch')
parser.add_argument('--patience', default=20000, type=int, help='patience for early stopping')
parser.add_argument('--verbose', action='store_true', help='verbose')
# DSAT add
parser.add_argument('--predict_turns_num', type=int, default=1, help='how turns for fix or predict')
parser.add_argument('--discriminator', type=bool, default=True, help='use discriminator or not')
parser.add_argument('--structure_feature', type=str, default='one-hot', help='one-hot or cosine-position-aware or position-aware')
parser.add_argument('--cuda', action='store_true', default=True, help='Use UDA training.')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--enc_name', type=str, default='GCN', help='Initial encoder model, GCN or GAT, Normal_GCN')
parser.add_argument('--r_type', type=str, default='flip', help='Attack Type: none/add/remove/flip')
parser.add_argument('--alpha', type=float, default=0.2, help='Initial alpha for leaky relu when use GAT as enc-name')
parser.add_argument('--loss_alpha_missing', type=float, default=0, help='missing in attributes')
parser.add_argument('--FP', type=int, default=20, help='0:FP times')
parser.add_argument('--lambda_gan', type=float, default=1, help='lambda for GAN loss, always 1.0 in our model')
parser.add_argument('--gan_alpha_missing', type=float, default=10, help='gan for missing attributes')
parser.add_argument('--seed', type=int, default=1024, help='Random seed.')
parser.add_argument('--not_change_adj', type=bool, default=True, help='Whether to change adjacency.')
parser.add_argument('--beta_2', type=int, default=200, help='decoupling coefficient.')
parser.add_argument('--use_lcc', action='store_true', default=True)
parser.add_argument('--position_aware_c', type=int, default=10, help='Size of position-aware data')
parser.add_argument('--split_train', type=bool, default=True, help='Whether to split fix and predict')
parser.add_argument('--common_z', type=bool, default=True, help='Whether to use common z in gan')
parser.add_argument('--use_fix', type=bool, default=True, help='Whether to fix')
parser.add_argument('--profile', type=bool, default=False, help='Experiment setting: node classification or profile')
parser.add_argument('--filter', type=str, default='Katz', help='Katz/Res/SGC/GCN/AGE')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if __name__ == '__main__':
    args.dataset = 'cora'
    args.type = 'uniform'
    args.rate = 0.9          # missing-rate
    args.verbose = 'store_true'
    args.attack_rate = 0     # attack-rate
    args.use_lcc = False

    '''
    参数表：
    '''
    if args.dataset in ['cora', 'citeseer']:
        args.nhid = 16
        args.FP = 20
        args.epoch = 10000
        args.structure_feature = 'one-hot'
        args.hidden = 16
        args.enc_name = 'GAT'
        args.not_change_adj = True
        args.common_z = True
        args.split_train = True
        args.seed = 1024
        args.filter = 'SGC'
    else:
        args.nhid = 64
        args.FP = 0
        args.epoch = 2000
        args.structure_feature = 'position-aware'
        args.hidden = 64
        args.enc_name = 'GCN'
        args.not_change_adj = True
        args.common_z = True
        args.split_train = True
        args.seed = 11024
        args.filter = 'Dense_Katz'

        args.beta_2 = 1000

    if args.type == 'struct':
        args.not_change_adj = False
    else:
        args.not_change_adj = True

    # data_prepare: build data-features\adj\label\tvt-mask\MASK
    # data=>adj, edge_list, features, labels, test_mask, train_mask, val_mask
    data = NodeClsData(args.dataset, args)

    # mask prepare
    mask = generate_mask(data.features, args.rate, args.type, args.profile, data)
    data.attributes_mask = mask
    data.ground_truth_feature = data.features
    data = DSAT_apply_mask(data, args.FP, args)

    # structure data prepare
    # structure_feature_path = r"structure_data/{}_{}_{}_structure_feature_{}_{}.npy".\
    #     format(args.seed, args.dataset, args.structure_feature, args.attack_rate, args.use_lcc)
    structure_feature_path = r"structure_data/{}_{}_{}_structure_feature_{}.npy".\
        format(args.seed, args.dataset, args.structure_feature, args.attack_rate)

    if os.path.exists(structure_feature_path):
        data.structure_data = torch.Tensor(np.load(structure_feature_path))
    else:
        data.structure_data = prepare_structure(data.adj, args.structure_feature, use_cuda=args.cuda)
        np.save(structure_feature_path, data.structure_data)

    # to device
    data.features = data.features.to(device)
    data.structure_data = data.structure_data.to(device)
    data.adj = data.adj.to(device)
    data.ground_truth_feature = data.ground_truth_feature.to(device)
    data.labels = data.labels.to(device)
    data.attributes_mask = data.attributes_mask.to(device)
    data.ground_truth_adj = data.adj.to(device)

    # define DSAT-model(model for predict)
    if args.structure_feature == 'position-aware':
        model_fix = ASD_VAE(n_nodes=data.adj.shape[0], n_fts=data.features.shape[1], n_hid=args.hidden, dropout=args.dropout, args=args, square=False, a_fts=data.structure_data.shape[1])
    else:
        model_fix = ASD_VAE(n_nodes=data.adj.shape[0], n_fts=data.features.shape[1], n_hid=args.hidden, dropout=args.dropout, args=args)

    # define model for fix
    model_predict = NormalGCN(data, nhid=args.nhid, dropout=args.dropout)

    params = {
        'lr': args.lr,
        'weight_decay': args.wd,
        'epochs': args.epoch,
        'patience': args.patience,
        'early_stopping': True
    }
    print(args)

    data.edge_list = data.edge_list.cpu()

    trainer = DSAT_NodeClsTrainer(data, model_fix, model_predict, params, args, niter=1, verbose=args.verbose)
    trainer.run()