import os
import time
from collections import defaultdict
from logging import getLogger

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.multi_dae import MultiDAE
from dataset import ClientsSampler, ClientsDataset, TestDataset
from metric import Recall_Precision_F1_OneCall_at_k_batch, NDCG_binary_at_k_batch, AUC_at_k_batch
from utils import get_datetime_str, ensure_dir, clip_norm_, sample_neighbor, get_upload_items, get_size


class Clients:
    def __init__(self, args, dataset):
        self.n_users = dataset.training_set[0].shape[0]
        self.n_items = dataset.n_items
        self.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
        self.model = MultiDAE(args, dataset)
        self.model.to(self.device)
        self.clients_data = ClientsDataset(dataset.training_set[0])
        self.enc_module_name = set(args.enc_module_name)
        self.dec_module_name = set(args.dec_module_name)
        self.protect_module_name = set(args.enc_module_name + args.dec_module_name)
        self.xi = args.xi
        self.rho = args.rho
        self.tau = args.tau
        self.first_iter = [True] * self.n_users
        self.fixed_items = {}
        self.items_candidate_for_rand = {}
        self.l1_norm_clip = args.l1_norm_clip
        self.lam = args.lam
        self.laplace = torch.distributions.Laplace(0, torch.tensor(self.lam, device=self.device))
        if args.perturb_method == 'MPC':
            self.perturb_fuc = self.MPC_perturb
        else:
            self.perturb_fuc = lambda g: g
        self.communication_cost = []
            
    def evaluate_restore(self, uid, x_pred):
        x_true = self.clients_data[uid].ravel()
        pre = precision_score(x_true, x_pred)
        recall = recall_score(x_true, x_pred)
        f1 = f1_score(x_true, x_pred)
        return pre, recall, f1

    def MPC_perturb(self, clients_grads):
        uids = np.array(list(clients_grads.keys()))
        clients_idx = sample_neighbor(len(uids), self.xi)
        clients_to_send = uids[clients_idx]
        communication_cost = 0.
        for i, uid in enumerate(uids):
            for grads_to_send in (clients_grads[suid] for suid in clients_to_send[i]):
                for name, grads in clients_grads[uid].items():
                    share = torch.randn_like(grads)
                    grads -= share
                    grads_to_send[name] += share
                    communication_cost += get_size(share)

        return clients_grads, communication_cost / (1 << 20)
    
    def count_update_communication_cost(self, clients_grads):
        communication_cost = 0.
        compressed = False
        for grads_dict in clients_grads.values():
            for name, grads in grads_dict.items():
                if compressed and name in self.protect_module_name:
                    dim = -1 if grads.shape[0] == self.n_items else 0
                    grad_mask = (torch.sum(grads, dim=dim, keepdim=True) != 0).float() if len(grads.shape) == 2 else (grads != 0).float()
                    para_percent = torch.sum(grad_mask).item() / self.n_items
                else:
                    para_percent = 1.
                communication_cost += get_size(grads) * para_percent
        return communication_cost / (1 << 20)
    
    def train(self, uids, model_param_state_dict):
        # receive model parameters from the server
        self.model.load_state_dict(model_param_state_dict)
        # count downloaded communications volume
        self.communication_cost.append(get_size(model_param_state_dict) * len(uids) / (1 << 20))
        x = []
        for uid in uids:
            x.append(self.clients_data[uid].view(1, -1))
        x = torch.cat(x, 0)
        x = x.to(self.device)
        # each client computes gradients using its private data
        clients_grads = {}
        for uid, x_u in zip(uids, x):
            x_u = x_u.view(1, -1)
            recon_batch = self.model(x_u)
            loss = self.model.loss_function(recon_batch, x_u)
            # self.optimizer.zero_grad(set_to_none=True)
            self.model.zero_grad(True)
            loss.backward()
            grad_u = {}
            for name, param in self.model.named_parameters():
                grad_u[name] = param.grad.detach().clone()
            clients_grads[uid.item()] = grad_u
        # perturb the original gradients
        perturb_grads, communication_cost = self.perturb_fuc(clients_grads)
        # count communications volume between clients
        self.communication_cost[-1] += communication_cost
        # send the gradients of each client to the server
        self.communication_cost[-1] += self.count_update_communication_cost(perturb_grads)
        self.communication_cost[-1] /= len(uids)
        return perturb_grads

    def get_mean_communications_cost(self):
        return sum(self.communication_cost) / max(len(self.communication_cost), 1)


class Server:
    def __init__(self, args, dataset, clients: Clients):
        self.logger = getLogger()
        self.seed = args.seed
        self.n_users = dataset.training_set[0].shape[0]
        self.n_items = dataset.n_items
        self.clients = clients
        self.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.batch_size = args.batch_size
        self.top_k = args.top_k
        self.valid_data = DataLoader(
            TestDataset(*dataset.validation_set),
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False)
        self.test_data = DataLoader(
            TestDataset(*dataset.test_set),
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=False)
        self.model = MultiDAE(args, dataset)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.enc_name = args.enc_module_name
        self.dec_name = args.dec_module_name
        self.use_enc_grad = args.use_enc_grad
        self.restore_epochs = args.restore_epochs
        
        logger_name = f'-{args.logger_name}' if args.logger_name else ''
        
        datetime = get_datetime_str()
        
        self.saved_path = os.path.join(args.saved_path, 'federated', args.model_name)
        ensure_dir(self.saved_path)
        self.saved_path = os.path.join(self.saved_path, f'{args.dataset_name}{logger_name}-{datetime}.pt')
        
        self.result_path = os.path.join(args.result_path, 'federated', args.model_name)
        ensure_dir(self.result_path)
        self.result_path = os.path.join(self.result_path, f'{args.dataset_name}{logger_name}-{datetime}.csv')
        
        self.use_tensorboard = args.use_tensorboard
        if self.use_tensorboard:
            self.tensorboard_path = os.path.join(args.tensorboard_path, 'federated', args.model_name, f'{args.dataset_name}{logger_name}-{datetime}')
            ensure_dir(self.tensorboard_path)
            self.writer = SummaryWriter(log_dir=self.tensorboard_path)

    def aggregate_gradients(self, clients_grads):
        clients_num = len(clients_grads)
        aggregated_gradients = defaultdict(float)
        for uid, grads_dict in clients_grads.items():
            for name, grad in grads_dict.items():
                aggregated_gradients[name] += grad / clients_num

        for name, param in self.model.named_parameters():
            if param.grad is None:
                param.grad = aggregated_gradients[name].detach().clone()
            else:
                param.grad += aggregated_gradients[name]
                
    def restore_from_gradient(self, clients_grads):
        res = []
        for uid, grads_dict in clients_grads.items():
            dec_grads = grads_dict[self.dec_name].cpu().numpy()
            euc_grads = None if self.enc_name == self.dec_name else grads_dict[self.enc_name].cpu().numpy()
            if dec_grads.shape[0] != self.n_items:
                dec_grads = dec_grads.T
            if euc_grads.shape[0] != self.n_items:
                euc_grads = euc_grads.T
            dec_grads_nonzero_idx = np.sum(dec_grads, axis=1).nonzero()[0]
            dec_grads_nonzero = dec_grads[dec_grads_nonzero_idx]
            kmeans = KMeans(n_clusters=2, random_state=self.seed)
            x_pred_ = kmeans.fit_predict(dec_grads_nonzero)
            select1 = kmeans.labels_.astype(bool)
            select0 = ~select1
            g_norm1 = np.linalg.norm(dec_grads_nonzero[select1], ord=2, axis=1).mean()
            g_norm0 = np.linalg.norm(dec_grads_nonzero[select0], ord=2, axis=1).mean()
            if g_norm1 < g_norm0:
                x_pred_ ^= 1
            x_pred = np.zeros(self.n_items)
            x_pred[dec_grads_nonzero_idx[x_pred_.astype(bool)]] = 1
            if euc_grads is not None and self.use_enc_grad:
                pos = np.sum(euc_grads, axis=1).nonzero()[0]
                x_pred[pos] = 1
            res.append(self.clients.evaluate_restore(uid, x_pred))
        res = np.array(res)
        self.logger.info(
            "Restoring result: Pre: {:5.4f} | Rec: {:5.4f} | F1: {:5.4f}".format(
                *np.mean(res, axis=0)
            ))

    def train(self):
        best_ndcg = -np.inf
        best_epoch = 0
        patience = self.early_stop
        for epoch in range(self.epochs):
            start = time.time()
            # train phase
            self.model.train()
            uid_seq = DataLoader(ClientsSampler(self.clients.n_users), batch_size=self.batch_size, shuffle=True)
            restored = False
            for uids in uid_seq:
                # sample clients to train the model
                self.optimizer.zero_grad(set_to_none=True)
                # send the model to the clients and let them start training
                clients_grads = self.clients.train(uids, self.model.state_dict())
                # restore only once in each restoring epoch
                if epoch + 1 in self.restore_epochs and not restored:
                    restored = True
                    self.restore_from_gradient(clients_grads)
                # aggregate the received gradients
                self.aggregate_gradients(clients_grads)
                # update the model
                self.optimizer.step()
            
            # log in tensorboard
            if self.use_tensorboard:
                self.log_in_tensorboard(epoch)
            
            # evaluate phase
            precision, recall, f1, ndcg, oneCAll, auc = self.evaluate(self.valid_data)

            self.logger.info(
                f"Epoch: {epoch + 1:3d} | Pre@{self.top_k}: {precision:5.4f} | Rec@{self.top_k}: {recall:5.4f} | F1@{self.top_k}: {f1:5.4f} |"
                f" NDCG@{self.top_k}: {ndcg:5.4f} | 1-call@{self.top_k}: {oneCAll:5.4f} | AUC: {auc:5.4f} | Time: {time.time() - start:5.4f}")

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_epoch = epoch + 1
                patience = self.early_stop
                self.logger.info(f'Save current model to [{self.saved_path}]')
                torch.save(self.model.state_dict(), self.saved_path)
            else:
                patience -= 1
                if patience == 0:
                    break
        self.logger.info("mean communication cost per round per user: {:.4f} MB".format(self.clients.get_mean_communications_cost()))
        self.logger.info(f'epoch of best ndcg@{self.top_k}({best_ndcg:5.4f}) is {best_epoch}')
        
    def evaluate(self, dataset, load_model=False):
        if load_model:
            self.model.load_state_dict(torch.load(self.saved_path))
        # evaluate phase
        ndcg_list = []
        recall_list = []
        precision_list = []
        f1_list = []
        oneCall_list = []
        auc_list = []

        self.model.eval()
        with torch.no_grad():
            for x, test_x in dataset:
                x = x.to(self.device)
                recon_batch = self.model(x)
                recon_batch = recon_batch.cpu().numpy()
                recon_batch[x.cpu().numpy().nonzero()] = -np.inf
                test_x = test_x.detach().numpy()
                n_k = NDCG_binary_at_k_batch(recon_batch, test_x, self.top_k)
                r_k, p_k, f_k, o_k = Recall_Precision_F1_OneCall_at_k_batch(recon_batch, test_x, self.top_k)
                auc_b = AUC_at_k_batch(x.cpu().numpy(), recon_batch, test_x)
                ndcg_list.append(n_k)
                recall_list.append(r_k)
                precision_list.append(p_k)
                f1_list.append(f_k)
                oneCall_list.append(o_k)
                auc_list.append(auc_b)

        ndcg_list = np.concatenate(ndcg_list)
        recall_list = np.concatenate(recall_list)
        precision_list = np.concatenate(precision_list)
        f1_list = np.concatenate(f1_list)
        oneCall_list = np.concatenate(oneCall_list)
        auc_list = np.concatenate(auc_list)

        ndcg_list[np.isnan(ndcg_list)] = 0
        ndcg = np.mean(ndcg_list)
        recall_list[np.isnan(recall_list)] = 0
        recall = np.mean(recall_list)
        precision_list[np.isnan(precision_list)] = 0
        precision = np.mean(precision_list)
        f1_list[np.isnan(f1_list)] = 0
        f1 = np.mean(f1_list)
        oneCall_list[np.isnan(oneCall_list)] = 0
        oneCAll = np.mean(oneCall_list)
        auc_list[np.isnan(auc_list)] = 0
        auc = np.mean(auc_list)

        return precision, recall, f1, ndcg, oneCAll, auc
    
    def test(self, save=False):
        precision, recall, f1, ndcg, oneCAll, auc = self.evaluate(self.test_data, True)

        res = f"Test: Pre@{self.top_k}: {precision:5.4f} | Rec@{self.top_k}: {recall:5.4f} | F1@{self.top_k}: {f1:5.4f} |"\
        f" NDCG@{self.top_k}: {ndcg:5.4f} | 1-call@{self.top_k}: {oneCAll:5.4f} | AUC: {auc:5.4f}"
        self.logger.info(res)
        
        if save:
            res_dt = dict([r.split(':') for r in res[6:].replace(' ', '').split('|')])
            df = pd.DataFrame(res_dt, index=[0])
            df.to_csv(self.result_path, sep='\t', index=False)
            self.logger.info(f'Result has been saved to [{self.result_path}]')
            
    def log_in_tensorboard(self, epoch):
        for name, parameter in self.model.named_parameters():
            self.writer.add_histogram(tag=f'{name}_data', 
                                      values=parameter,
                                      global_step=epoch
                                      )
