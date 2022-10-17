"""
version 1.0
date 2021/02/04
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import numpy as np
import torch
import torch.nn.functional as F

from numpy import mean, std
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import Adam
from tqdm import tqdm
from utils import plot_TSNE
from sklearn.utils import shuffle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    def __init__(self, patience, verbose):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.state_dict = None

    def reset(self):
        self.counter = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.state_dict = None

    def check(self, evals, model, epoch):
        if evals['val_loss'] <= self.best_val_loss or evals['val_acc'] >= self.best_val_acc:
            if evals['val_loss'] <= self.best_val_loss and evals['val_acc'] >= self.best_val_acc:
                # self.state_dict = deepcopy(model.state_dict())
                pass
            self.best_val_loss = min(self.best_val_loss, evals['val_loss'])
            self.best_val_acc = max(self.best_val_acc, evals['val_acc'])
            self.counter = 0
        else:
            self.counter += 1
        stop = False
        if self.counter >= self.patience:
            stop = True
            if self.verbose:
                print("Stop training, epoch:", epoch)
            # model.load_state_dict(self.state_dict)
        return stop


class NodeClsTrainer:
    def __init__(self, data, model_fix, model_predict, params, args, niter=100, verbose=False):
        self.data = data
        self.args = args

        self.model_fix = model_fix.to(device)
        self.model_predict = model_predict.to(device)
        self.optimizer_fix = Adam(model_fix.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        self.optimizer_predict = Adam(model_predict.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

        self.optimizer_common = torch.optim.SGD([{'params': model_fix.parameters(), 'lr':params['lr'], 'weight_decay':params['weight_decay']},
                                     {'params': model_predict.parameters(), 'lr': params['lr']}],
                                    lr=params['lr'], momentum=0.9)

        self.lr = params['lr']
        self.weight_decay = params['weight_decay']
        self.epochs = params['epochs']
        self.niter = niter
        self.verbose = verbose
        self.early_stopping = params['early_stopping']
        if self.early_stopping:
            self.stop_checker = EarlyStopping(params['patience'], verbose)

        self.data.to(device)
        self.BCE = torch.nn.BCEWithLogitsLoss(reduction='none')

    def reset(self):
        self.model_predict.to(device).reset_parameters()
        self.optimizer_predict = Adam(self.model_predict.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.early_stopping:
            self.stop_checker.reset()

    def train(self, token, ff):
        data, model_predict, model_fix, optimizer_predict, optimizer_fix, optimizer_common = \
            self.data, self.model_predict, self.model_fix, self.optimizer_predict, self.optimizer_fix, self.optimizer_common

        prior = torch.distributions.normal.Normal(loc=torch.FloatTensor([0.0]), scale=torch.FloatTensor([1.0]))

        indices = torch.arange(0, data.adj.shape[0]).to(device)
        train_fts_idx_arr = torch.masked_select(indices, data.train_mask)

        pos_weight = torch.sum(data.features[train_fts_idx_arr] == 0.0).item() / (
            torch.sum(data.features[train_fts_idx_arr] != 0.0).item())

        weight_mask = torch.where(self.data.features != 0.0,
                                  pos_weight*torch.ones_like(self.data.features), torch.ones_like(self.data.features))

        missing_alpha_mask = (data.attributes_mask==True) * self.args.loss_alpha_missing + (data.attributes_mask==False)
        if not self.args.use_fix:
            token = "predict"
        self.common_train_loss = 0

        if token == "fix":
            model_fix.train()
            if self.args.split_train:
                optimizer_fix.zero_grad()
            else:
                optimizer_common.zero_grad()
                self.common_train_loss = 0

            torch.cuda.empty_cache()
            x_pre, a_pre = model_fix(data.features, data.adj, data.structure_data)
            torch.cuda.empty_cache()
            # compute loss
            if self.args.not_change_adj:
                a_pre = a_pre.detach()
                fix_loss = model_fix.loss(data.features, data.adj, train_fts_idx_arr, weight_mask,
                                            missing_alpha_mask, beta_2=self.args.beta_2)
            else:
                gamma = 1
                a_new_pre = a_pre.detach()
                data.adj = (gamma * a_new_pre + data.ground_truth_adj).to_sparse()
                fix_loss = model_fix.loss(data.features, data.ground_truth_adj, train_fts_idx_arr, weight_mask,
                                            missing_alpha_mask, beta_2=self.args.beta_2, a_pre_true=a_new_pre)

            x_new_pre = x_pre.detach()
            data.features = data.ground_truth_feature * (data.attributes_mask==False) + 0.8 * x_new_pre * \
                            (data.attributes_mask==True) + 0.2 * data.features * (data.attributes_mask==True)
            if self.args.discriminator:
                torch.cuda.empty_cache()
                model_fix.train()
                if self.args.split_train:
                    optimizer_fix.zero_grad()
                else:
                    optimizer_common.zero_grad()
                x_pre, a_pre = model_fix(data.features, data.adj, data.structure_data)

                if self.args.common_z:
                    z_x = model_fix.q_Z[train_fts_idx_arr]
                else:
                    z_x = model_fix.z_x[train_fts_idx_arr]

                # sample gaussian noise
                true_z = prior.sample([z_x.shape[0], z_x.shape[1]]).reshape([z_x.shape[0], z_x.shape[1]])
                if self.args.cuda:
                    true_z = true_z.cuda()

                tmp_zx = model_fix.disc(true_z)

                tmp_zx_missing_attributes = tmp_zx * (data.attributes_mask[train_fts_idx_arr]==True)
                true_logits_zx = tmp_zx.reshape([-1])

                tmp_fake_logits_zx = model_fix.disc(z_x)

                tmp_fake_logits_zx_missing = tmp_fake_logits_zx * (data.attributes_mask[train_fts_idx_arr]==True)
                fake_logits_zx = tmp_fake_logits_zx.reshape([-1])
                logits_zx = torch.cat([true_logits_zx, fake_logits_zx])
                D_lable_10 = torch.cat([torch.ones_like(true_logits_zx), torch.zeros_like(fake_logits_zx)])
                D_loss = self.args.lambda_gan * self.BCE(logits_zx, D_lable_10).mean()

                missing_zx = tmp_zx_missing_attributes[tmp_zx_missing_attributes!=0]
                missing_fake_zx = tmp_fake_logits_zx_missing[tmp_fake_logits_zx_missing!=0]
                missing_logits_zx = torch.cat([missing_zx, missing_fake_zx])
                missing_D_lable_10 = torch.cat([torch.ones_like(missing_zx), torch.zeros_like(missing_fake_zx)])
                D_loss += self.args.lambda_gan * self.BCE(missing_logits_zx, missing_D_lable_10).mean() * self.args.gan_alpha_missing

                torch.nn.utils.clip_grad_norm_(parameters=model_fix.parameters(), max_norm=10, norm_type=2)

                fix_loss += D_loss

            if self.args.split_train:
                torch.cuda.empty_cache()
                fix_loss.backward()
                torch.cuda.empty_cache()
                optimizer_fix.step()
            else:
                self.common_train_loss += fix_loss
                pass

        elif token == "predict":
            model_predict.train()
            if self.args.split_train:
                optimizer_predict.zero_grad()
            else:
                if self.common_train_loss == 0:
                    optimizer_common.zero_grad()
                pass
            output = model_predict(data)
            if ff:
                plot_TSNE('cora', 'MDVAE', data.test_mask, model_predict.embedding, data.labels)
            loss_predict = F.nll_loss(output[data.train_mask], data.labels[data.train_mask])  # 求解loss损失

            if self.args.split_train:
                torch.cuda.empty_cache()
                loss_predict.backward()
                torch.cuda.empty_cache()
                optimizer_predict.step()
            else:
                torch.cuda.empty_cache()
                self.common_train_loss += loss_predict
                self.common_train_loss.backward()
                torch.cuda.empty_cache()
                optimizer_common.step()

    def evaluate(self):
        data, model_predict, model_fix = self.data, self.model_predict, self.model_fix
        model_predict.eval()
        model_fix.eval()

        with torch.no_grad():
            output = model_predict(data)

        outputs = {}
        for key in ['train', 'val', 'test']:
            if key == 'train':
                mask = data.train_mask
            elif key == 'val':
                mask = data.val_mask
            else:
                mask = data.test_mask
            loss = F.nll_loss(output[mask], data.labels[mask]).item()
            pred = output[mask].max(dim=1)[1]
            acc = pred.eq(data.labels[mask]).sum().item() / mask.sum().item()

            outputs['{}_loss'.format(key)] = loss
            outputs['{}_acc'.format(key)] = acc

        return outputs

    def print_verbose(self, epoch, evals):
        print('epoch: {: 5d}'.format(epoch),
              'train loss: {:.5f}'.format(evals['train_loss']),
              'train acc: {:.5f}'.format(evals['train_acc']),
              'val loss: {:.5f}'.format(evals['val_loss']),
              'val acc: {:.5f}'.format(evals['val_acc']))

    def RECALL_NDCG(self, estimated_fts, true_fts, topN=10, setting=False, mask=None):
        if setting:
            indices = torch.arange(0, estimated_fts.shape[0])
            test_indice = torch.masked_select(indices, mask)
            estimated_fts = estimated_fts[test_indice]
            true_fts = true_fts[test_indice]

        preds = np.argsort(-estimated_fts, axis=1)
        preds = preds[:, :topN]

        gt = [np.where(true_fts[i, :] != 0)[0] for i in range(true_fts.shape[0])]
        recall_list = []
        ndcg_list = []
        for i in range(preds.shape[0]):
            # calculate recall
            if len(gt[i]) != 0:

                # whether the generated feature is non feature
                if np.sum(estimated_fts[i, :]) != 0:
                    recall = len(set(preds[i, :]) & set(gt[i])) * 1.0 / len(set(gt[i]))
                    recall_list.append(recall)

                    # calculate ndcg
                    intersec = np.array(list(set(preds[i, :]) & set(gt[i])))
                    if len(intersec) > 0:
                        dcg = [np.where(preds[i, :] == ele)[0] for ele in intersec]
                        dcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in dcg])
                        idcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in range(len(gt[i]))])
                        ndcg = dcg * 1.0 / idcg
                    else:
                        ndcg = 0.0
                    ndcg_list.append(ndcg)
                else:
                    temp_preds = shuffle(np.arange(estimated_fts.shape[1]))[:topN]

                    recall = len(set(temp_preds) & set(gt[i])) * 1.0 / len(set(gt[i]))
                    recall_list.append(recall)

                    # calculate ndcg
                    intersec = np.array(list(set(temp_preds) & set(gt[i])))
                    if len(intersec) > 0:
                        dcg = [np.where(temp_preds == ele)[0] for ele in intersec]
                        dcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in dcg])
                        idcg = np.sum([1.0 / (np.log2(x + 1 + 1)) for x in range(len(gt[i]))])
                        ndcg = dcg * 1.0 / idcg
                    else:
                        ndcg = 0.0
                    ndcg_list.append(ndcg)

        avg_recall = np.mean(recall_list)
        avg_ndcg = np.mean(ndcg_list)

        return avg_recall, avg_ndcg

    def run(self):
        val_acc_list = []
        test_acc_list = []

        for _ in tqdm(range(self.niter)):
            self.reset()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            best_val_acc = 0.
            best_eval = 0.
            for epoch in range(1, self.epochs + 1):
                ff = False
                if epoch == self.epochs-1:
                    ff = False
                if self.args.split_train:
                    if epoch % (self.args.predict_turns_num+1) == 0 or epoch > self.epochs/2:
                        token = "predict"
                    else:
                        token = "fix"
                else:
                    if epoch % (self.args.predict_turns_num + 1) == 0 or epoch > self.epochs / 2:
                        token = "predict"
                    else:
                        token = "fix"

                self.train(token, ff)
                evals = self.evaluate()

                if self.verbose:
                    self.print_verbose(epoch, evals)

                if self.early_stopping:
                    if self.stop_checker.check(evals, self.model_predict, epoch):
                        break
                if evals['val_acc'] > best_val_acc:
                    best_val_acc = evals['val_acc']
                    best_eval = evals

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            evals = self.evaluate()
            if self.verbose:
                print("best result:", end=' ')
                for met, val in best_eval.items():
                    print(met, val)

            val_acc_list.append(best_eval['val_acc'])
            test_acc_list.append(best_eval['test_acc'])

        print(mean(test_acc_list))
        print(std(test_acc_list))


        if self.args.dataset in ['cora', 'citeseer', 'amaphoto', 'amacomp']:
            '''
            evaluation for Recall and NDCG
            '''
            mask = self.data.test_mask
            for topK in [10, 20, 50]:
                avg_recall, avg_ndcg = self.RECALL_NDCG(self.data.features.cpu().numpy(),
                                                        self.data.ground_truth_feature.cpu().numpy(), topK,
                                                        self.args.profile, mask.cpu())
                print('tpoK: {}, recall: {}, ndcg: {}'.format(topK, avg_recall, avg_ndcg))
            print('dataset: {}'.format(self.args.dataset))

        return {
            'val_acc': mean(val_acc_list),
            'test_acc': mean(test_acc_list),
            'test_acc_std': std(test_acc_list)
        }


def reconstruction_loss(data, output):
    adj_recon = output['adj_recon']
    return data.norm * F.binary_cross_entropy_with_logits(adj_recon, data.adjmat, pos_weight=data.pos_weight)


def linkpred_loss(data, output):
    recon_loss = reconstruction_loss(data, output)
    mu, logvar = output['mu'], output['logvar']
    kl = - 1 / (2 * data.num_nodes) * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return recon_loss + kl


def linkpred_score(z, pos_edges, neg_edges):
    pos_score = torch.sigmoid(torch.sum(z[pos_edges[0]] * z[pos_edges[1]], dim=1))
    neg_score = torch.sigmoid(torch.sum(z[neg_edges[0]] * z[neg_edges[1]], dim=1))
    pred_score = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    true_score = np.hstack([np.ones(pos_score.size(0)), np.zeros(neg_score.size(0))])
    auc_score = roc_auc_score(true_score, pred_score)
    ap_score = average_precision_score(true_score, pred_score)
    return auc_score, ap_score
