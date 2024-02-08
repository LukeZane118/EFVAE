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

from model.ef_multi_vae import EFMultiVAE
from sampler import get_sampler
from dataset import ClientsSampler, EFClientsDataset, TestDataset
from metric import Recall_Precision_F1_OneCall_at_k_batch, NDCG_binary_at_k_batch, AUC_at_k_batch
from utils import get_datetime_str, ensure_dir, clip_norm_, sample_neighbor, get_upload_items, get_size
from sampler.base import SamplerBase


class Clients:
    def __init__(self, args, dataset):
        self.n_users = dataset.training_set[0].shape[0]
        self.n_items = dataset.n_items
        self.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
        self.model = EFMultiVAE(args, dataset)
        self.model.to(self.device)
        self.clients_data = EFClientsDataset(dataset.training_set[0])
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
            self.perturb_fuc = lambda g, s: (g, 0)
        self.communication_cost = [] 
            
    def evaluate_restore(self, uid, x_pred):
        x_true = self.clients_data[uid].ravel()
        pre = precision_score(x_true, x_pred)
        recall = recall_score(x_true, x_pred)
        f1 = f1_score(x_true, x_pred)
        return pre, recall, f1

    @torch.no_grad()
    def MPC_perturb(self, clients_grads, sampling_items):
        if self.xi == 0:
            return clients_grads, 0
        uids = np.array(list(clients_grads.keys()))
        sent_items = np.zeros([len(uids), self.n_items], dtype=np.int_)
        for i, uid in enumerate(uids):
            fake_items_u = self.get_fake_items(uid)
            true_items_u = self.clients_data[uid].nonzero(as_tuple=True)[0].numpy()
            hybrid_items_u = np.union1d(fake_items_u, true_items_u)
            sampling_items_u = sampling_items[i]
            final_items_u = np.union1d(hybrid_items_u, sampling_items_u)
            sent_items[i, final_items_u] = 1

        # record the users to be sent and the items to be uploaded
        clients_idx = sample_neighbor(len(uids), self.xi)
        clients_to_send = uids[clients_idx]
        
        communication_cost = 0.
        
        # send gradients
        for i, uid in enumerate(uids):
            noise_mask = ~sent_items[i].astype(bool)
            # percentage of parameters to send
            send_percent = np.sum(sent_items[i]) / self.n_items
            for name, grads in clients_grads[uid].items():
                # space exchange time to speed up
                share = torch.randn((len(clients_to_send[i]), *grads.shape), device=self.device)
                if name in self.protect_module_name:
                    share[:, noise_mask] = 0.
                    communication_cost += get_size(share) * send_percent
                else:
                    communication_cost += get_size(share)
                    
                grads -= share.sum(0)
                for j, grads_to_send in enumerate(clients_grads[suid] for suid in clients_to_send[i]):
                    # for the convenience of programming, we send the entire tensor, i.e., random_nums,
                    # but actually only the non-zero value of it needs to be sent
                    grads_to_send[name] += share[j]
                    
        return clients_grads, communication_cost / (1 << 20)
    
    def get_fake_items(self, uid):
        # fixed items for protecting interaction behaviors
        if uid not in self.fixed_items:
            i_u = self.clients_data[uid].nonzero(as_tuple=True)[0].numpy()
            item_candidate = np.setdiff1d(np.arange(self.n_items), i_u, assume_unique=True)
            num_fixed = int(min(self.rho * i_u.shape[0], item_candidate.shape[0]))
            fixed_items = np.random.choice(item_candidate, num_fixed, replace=False)
            self.fixed_items[uid] = fixed_items
        else:
            fixed_items = self.fixed_items[uid]
        return fixed_items
    
    def train(self, uids: torch.Tensor, model_param_state_dict: dict, sampler: SamplerBase, anneal: float) -> dict:
        # receive model parameters from the server
        self.model.load_state_dict(model_param_state_dict)

        # each client computes gradients using its private data
        clients_grads = {}
        self.communication_cost.append(0)
        all_sampling_items = []
        for uid in uids:
            pos_items = self.clients_data[uid.item()].view(1, -1).to(self.device)
            loss, sampling_items = self.model.calculate_loss(pos_items, sampler, anneal)
            self.model.zero_grad(True)
            loss.backward()
            grad_u = {}
            for name, param in self.model.named_parameters():
                grad_u[name] = param.grad.detach().clone() if param.grad is not None else None
            clients_grads[uid.item()] = grad_u
            all_sampling_items.append(sampling_items.detach().cpu().numpy())
            
        self.communication_cost[-1] += self.count_download_communication_cost(uids.detach().cpu().numpy(), all_sampling_items, model_param_state_dict, sampler)
                
        # perturb the original gradients
        perturb_grads, communication_cost = self.perturb_fuc(clients_grads, all_sampling_items)
        self.communication_cost[-1] += communication_cost
        self.communication_cost[-1] += self.count_update_communication_cost(clients_grads)
        self.communication_cost[-1] /= len(uids)
        # send the gradients of each client to the server
        return perturb_grads
    
    @torch.no_grad()
    def count_download_communication_cost(self, uids, sampling_items, model_param_state_dict, sampler: SamplerBase):
        if not hasattr(self, "full_enc_communication_cost"):
            self.full_enc_communication_cost = sum(get_size(model_param_state_dict[name]) for name in self.enc_module_name) / (1 << 20)
        if not hasattr(self, "full_dec_communication_cost"):
            self.full_dec_communication_cost = sum(get_size(model_param_state_dict[name]) for name in self.dec_module_name) / (1 << 20)
        if not hasattr(self, "other_communication_cost"):
            self.other_communication_cost = sum(get_size(param) for name, param in model_param_state_dict.items() if name not in self.protect_module_name) / (1 << 20) + \
                sampler.get_communication_cost()
            
        communication_cost = 0.
        for i, uid in enumerate(uids):
            fake_items_u = self.get_fake_items(uid)
            true_items_u = self.clients_data[uid].nonzero(as_tuple=True)[0].numpy()
            hybrid_items_u = np.union1d(fake_items_u, true_items_u)
            sampling_items_u = sampling_items[i]
            final_items_u = np.union1d(hybrid_items_u, sampling_items_u)
            para_percent_enc = len(hybrid_items_u) / self.n_items
            para_percent_dec = len(final_items_u) / self.n_items
            communication_cost += self.full_enc_communication_cost * para_percent_enc + self.full_dec_communication_cost * para_percent_dec
        communication_cost += self.other_communication_cost * len(uids)
            
        return communication_cost
    
    @torch.no_grad()
    def count_update_communication_cost(self, clients_grads):
        communication_cost = 0.
        for grads_dict in clients_grads.values():
            for name, grads in grads_dict.items():
                if name in self.protect_module_name:
                    grad_mask = (torch.sum(grads, dim=-1, keepdim=True) != 0).float() if len(grads.shape) == 2 else (grads != 0).float()
                    para_percent = torch.sum(grad_mask).item() / self.n_items
                else:
                    para_percent = 1.
                communication_cost += get_size(grads) * para_percent
        return communication_cost / (1 << 20)
    
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
        self.total_anneal_steps = args.total_anneal_steps
        self.anneal_cap = args.anneal_cap
        self.batch_size = args.batch_size
        self.update_count = 0
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
        self.model = EFMultiVAE(args, dataset)
        self.model.to(self.device)
        self.sampler = get_sampler(args, dataset)
        self.sampler.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.enc_name = args.enc_module_name
        self.dec_name = args.dec_module_name
        self.use_enc_grad = args.use_enc_grad
        self.restore_epochs = args.restore_epochs
        self.target_module_name = set(args.enc_module_name + args.dec_module_name)
        
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
        # TODO: Fixed mean
        # clients_num = len(clients_grads)
        aggregated_gradients = defaultdict(float)
        gradient_count = defaultdict(int)
        for uid, grads_dict in clients_grads.items():
            for name, grads in grads_dict.items():
                if grads is not None:
                    aggregated_gradients[name] += grads
                    if name in self.target_module_name:
                        dim = -1 if grads.shape[0] == self.n_items else 0
                        grad_mask = (torch.sum(grads, dim=dim, keepdim=True) != 0).float() if len(grads.shape) == 2 else (grads != 0).float()
                        gradient_count[name] += grad_mask
                    else:
                        gradient_count[name] += 1

        for name, param in self.model.named_parameters():
            if aggregated_gradients[name] is None:
                continue
            if name in self.target_module_name:
                gradient_count[name] += 1e-8
            if param.grad is None:
                param.grad = aggregated_gradients[name].detach().clone() / gradient_count[name]
            else:
                param.grad += aggregated_gradients[name] / gradient_count[name]
                
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
            if hasattr(self.sampler, "update"):
                # self.sampler.update(self.model.dec_embeddings.weight, self.model.dec_bias.weight)
                self.sampler.update(self.model.dec_embeddings.weight)
            for uids in uid_seq:
                # sample clients to train the model
                if self.total_anneal_steps > 0:
                    anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
                else:
                    anneal = self.anneal_cap
                self.update_count += 1
                self.optimizer.zero_grad(set_to_none=True)
                # send the model to the clients and let them start training
                clients_grads = self.clients.train(uids, self.model.state_dict(), self.sampler, anneal)
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
            precision5, recall5, f1, ndcg5, oneCAll, auc = self.evaluate(self.valid_data)

            self.logger.info(
                "Epoch: {:3d} | Pre@5: {:5.4f} | Rec@5: {:5.4f} | F1@5: {:5.4f} | NDCG@5: {:5.4f} | 1-call@5: {:5.4f} | AUC: {:5.4f} | Time: {:5.4f}".format(
                    epoch + 1, precision5, recall5, f1, ndcg5, oneCAll, auc, time.time() - start))

            if ndcg5 > best_ndcg:
                best_ndcg = ndcg5
                best_epoch = epoch + 1
                patience = self.early_stop
                self.logger.info(f'Save current model to [{self.saved_path}]')
                torch.save(self.model.state_dict(), self.saved_path)
            else:
                patience -= 1
                if patience == 0:
                    break
        self.logger.info('epoch of best ndcg@5({:5.4f}) is {}'.format(best_ndcg, best_epoch))
        self.logger.info("mean communication cost per round per user: {:.4f} MB".format(self.clients.get_mean_communications_cost()))
        
    def evaluate(self, dataset, load_model=False):
        if load_model:
            self.model.load_state_dict(torch.load(self.saved_path))
        # evaluate phase
        ndcg5_list = []
        recall5_list = []
        precision5_list = []
        f1_list = []
        oneCall_list = []
        auc_list = []

        self.model.eval()
        with torch.no_grad():
            for x, test_x in dataset:
                x = x.to(self.device)
                recon_batch, mu, logvar = self.model(x)
                recon_batch = recon_batch.cpu().numpy()
                recon_batch[x.cpu().numpy().nonzero()] = -np.inf
                test_x = test_x.detach().numpy()
                n_5 = NDCG_binary_at_k_batch(recon_batch, test_x, 5)
                r_5, p_5, f_5, o_5 = Recall_Precision_F1_OneCall_at_k_batch(recon_batch, test_x, 5)
                auc_b = AUC_at_k_batch(x.cpu().numpy(), recon_batch, test_x)
                ndcg5_list.append(n_5)
                recall5_list.append(r_5)
                precision5_list.append(p_5)
                f1_list.append(f_5)
                oneCall_list.append(o_5)
                auc_list.append(auc_b)

        ndcg5_list = np.concatenate(ndcg5_list)
        recall5_list = np.concatenate(recall5_list)
        precision5_list = np.concatenate(precision5_list)
        f1_list = np.concatenate(f1_list)
        oneCall_list = np.concatenate(oneCall_list)
        auc_list = np.concatenate(auc_list)

        ndcg5_list[np.isnan(ndcg5_list)] = 0
        ndcg5 = np.mean(ndcg5_list)
        recall5_list[np.isnan(recall5_list)] = 0
        recall5 = np.mean(recall5_list)
        precision5_list[np.isnan(precision5_list)] = 0
        precision5 = np.mean(precision5_list)
        f1_list[np.isnan(f1_list)] = 0
        f1 = np.mean(f1_list)
        oneCall_list[np.isnan(oneCall_list)] = 0
        oneCAll = np.mean(oneCall_list)
        auc_list[np.isnan(auc_list)] = 0
        auc = np.mean(auc_list)

        return precision5, recall5, f1, ndcg5, oneCAll, auc
    
    def test(self, save=False):
        precision5, recall5, f1, ndcg5, oneCAll, auc = self.evaluate(self.test_data, True)

        res = "Test: Pre@5: {:5.4f} | Rec@5: {:5.4f} | F1@5: {:5.4f} | NDCG@5: {:5.4f} | 1-call@5: {:5.4f} | AUC: {:5.4f}".format(
                precision5, recall5, f1, ndcg5, oneCAll, auc)
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
