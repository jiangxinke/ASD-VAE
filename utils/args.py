import os
import configparser


def add_args(parser, data_set="cora", type='uniform', missing_rate=0.1):
    # Data set settings.  HZY_west, HZY_east, PEMS03, PEMS04
    parser.add_argument('--dataset',
                        default=data_set,
                        choices=['cora', 'citeseer', 'amacomp', 'amaphoto'],
                        help='dataset name')
    parser.add_argument('--type',
                        default=type,
                        choices=['uniform', 'bias', 'struct'],
                        help="uniform randomly missing, biased randomly missing, and structurally missing")
    parser.add_argument('--rate', default=missing_rate, type=float, help='missing rate')
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--predict_turns_num', type=int, default=1, help='how turns for fix or predict')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use UDA training.')
    parser.add_argument('--lambda_gan', type=float, default=1, help='lambda for GAN loss, always 1.0 in our model')
    parser.add_argument('--use_lcc', action='store_true', default=False)
    parser.add_argument('--profile', type=bool, default=False,
                        help='Experiment setting: node classification or profile')
    parser.add_argument('--use_fix', type=bool, default=True, help='Whether to fix')
    parser.add_argument('--split_train', type=bool, default=True, help='Whether to split fix and predict')
    parser.add_argument('--position_aware_c', type=int, default=10, help='Size of position-aware data')
    parser.add_argument('--FP', type=int, default=10, help='0:FP times')
    parser.add_argument('--loss_alpha_missing', type=float, default=0, help='missing in attributes')
    parser.add_argument('--r_type', type=str, default='flip', help='Attack Type: none/add/remove/flip')
    parser.add_argument('--common_z', type=bool, default=True, help='Whether to use common z in gan')

    config_path = r'..\PEMS\{}.conf'.format(data_set)
    if os.path.exists(config_path):
        load_paras_config(parser, config_path)
    else:
        # Train
        parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
        parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')
        parser.add_argument('--dropout', default=0.4, type=float, help='dropout rate')
        parser.add_argument('--epoch', default=2000, type=int, help='the number of training epoch')
        parser.add_argument('--patience', default=500, type=int, help='patience for early stopping')
        parser.add_argument('--seed', type=int, default=1024, help='Random seed.')
        parser.add_argument('--attack_rate', default=0, type=float, help='attack rate in structure data')
        # Model
        parser.add_argument('--structure_feature', type=str, default='one-hot',
                            help='one-hot or cosine-position-aware or position-aware')
        parser.add_argument('--nhid', default=64, type=int, help='the number of hidden units in predictor')
        parser.add_argument('--discriminator', type=bool, default=True, help='use discriminator or not')
        parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units in GE.')
        parser.add_argument('--enc_name', type=str, default='GCN', help='Initial encoder model, GCN or GAT, Normal_GCN')
        parser.add_argument('--alpha', type=float, default=0.2,
                            help='Initial alpha for leaky relu when use GAT as enc-name')
        parser.add_argument('--gan_alpha_missing', type=float, default=10, help='gan for missing attributes')
        parser.add_argument('--not_change_adj', type=bool, default=True, help='Struct Update.')
        parser.add_argument('--beta_2', type=int, default=200, help='decoupling coefficient.')
        parser.add_argument('--filter', type=str, default='Katz', help='Katz/Res/SGC/GCN/AGE')


def load_paras_config(parser, config_path):
    # get configuration
    config = configparser.ConfigParser()
    config.read(config_path)
    print("++++++++++++++")
    print(config_path)  # using PEMS_{}.conf as "config"!
    print("++++++++++++++")
    # hyper
    parser.add_argument('--lr', default=config['hyper']['lr'], type=float)
    parser.add_argument('--wd', default=config['hyper']['wd'], type=float)
    parser.add_argument('--dropout', default=config['hyper']['dropout'], type=float)
    parser.add_argument('--epoch', default=config['hyper']['epoch'], type=int)
    parser.add_argument('--patience', default=config['hyper']['patience'], type=int)
    parser.add_argument('--seed', default=config['hyper']['seed'], type=int)
    parser.add_argument('--attack_rate', default=config['hyper']['attack_rate'], type=float)
    # model
    parser.add_argument('--structure_feature', default=config['Model']['structure_feature'], type=str)
    parser.add_argument('--nhid', default=config['Model']['nhid'], type=int)
    parser.add_argument('--discriminator', default=config['Model']['discriminator'], type=bool)
    parser.add_argument('--hidden', default=config['Model']['hidden'], type=int)
    parser.add_argument('--enc_name', default=config['Model']['enc_name'], type=str, help="GCN")
    parser.add_argument('--alpha', default=config['Model']['alpha'], type=float, help="alpha in GAT")
    parser.add_argument('--gan_alpha_missing', default=config['data']['gan_alpha_missing'], type=float)
    parser.add_argument('--not_change_adj', default=config['data']['not_change_adj'], type=bool)
    parser.add_argument('--beta_2', default=config['data']['beta_2'], type=int)
    parser.add_argument('--filter', default=config['data']['filter'], type=str)