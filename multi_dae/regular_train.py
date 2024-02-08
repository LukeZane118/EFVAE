import os
import time
from logging import getLogger

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.multi_dae import MultiDAE
from dataset import TrainDataset, TestDataset
from metric import Recall_Precision_F1_OneCall_at_k_batch, NDCG_binary_at_k_batch, AUC_at_k_batch
from utils import get_datetime_str, ensure_dir


class Trainer(object):
    def __init__(self, args, dataset):
        self.logger = getLogger()
        self.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
        self.epochs = args.epochs
        self.early_stop = args.early_stop
        self.batch_size = args.batch_size
        self.train_data = DataLoader(
            TrainDataset(dataset.training_set[0]),
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=True)
        self.valid_data = DataLoader(
            TestDataset(*dataset.validation_set),
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False)
        self.test_data = DataLoader(
            TestDataset(*dataset.test_set),
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False)
        self.model = MultiDAE(args, dataset)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        logger_name = f'-{args.logger_name}' if args.logger_name else ''
        
        datetime = get_datetime_str()
        
        self.saved_path = os.path.join(args.saved_path, 'regular', args.model_name)
        ensure_dir(self.saved_path)
        self.saved_path = os.path.join(self.saved_path, f'{args.dataset_name}{logger_name}-{datetime}.pt')
        
        self.result_path = os.path.join(args.result_path, 'regular', args.model_name)
        ensure_dir(self.result_path)
        self.result_path = os.path.join(self.result_path, f'{args.dataset_name}{logger_name}-{datetime}.csv')
        
        self.use_tensorboard = args.use_tensorboard
        if self.use_tensorboard:
            self.tensorboard_path = os.path.join(args.tensorboard_path, 'regular', args.model_name, f'{args.dataset_name}{logger_name}-{datetime}')
            ensure_dir(self.tensorboard_path)
            self.writer = SummaryWriter(log_dir=self.tensorboard_path)

    def train(self):
        best_ndcg = -np.inf
        best_epoch = 0
        patience = self.early_stop
        for epoch in range(self.epochs):
            start = time.time()
            # train phase
            self.model.train()
            for x_u in self.train_data:
                # sample clients to train the model
                x_u = x_u.to(self.device)
                recon_batch = self.model(x_u)
                loss = self.model.loss_function(recon_batch, x_u)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
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
                recon_batch = self.model(x)
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