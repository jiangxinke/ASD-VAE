import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
from models import NormalGCN, ASD_VAE
from train import NodeClsTrainer
from utils import NodeClsData, generate_mask, prepare_structure, ASD_apply_mask, add_args
import random
import numpy as np
import warnings

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Add Args
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    args.verbose = 'store_true'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    data = ASD_apply_mask(data, args.FP, args)

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

    trainer = NodeClsTrainer(data, model_fix, model_predict, params, args, niter=1, verbose=args.verbose)
    trainer.run()