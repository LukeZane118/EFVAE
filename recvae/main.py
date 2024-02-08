import os
import argparse
from logging import getLogger
import sys

# root_path = os.path.dirname(os.getcwd())
sys.path.insert(1, os.getcwd())

from utils import init_seed, init_logger, log_args
from dataset import load_data
from regular_train import Trainer
from federated_train import Server, Clients
import torch

# os.chdir(root_path) # change to root path

def parse_args():
    parser = argparse.ArgumentParser()
    
    # environment setting
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('-rp', '--reproducibility', default=True, help='Is regular training', action='store_false')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--state', type=str, default='info', help='Logging event level')
    parser.add_argument('-mn', '--model_name', type=str, default='recvae', help='Model name')
    parser.add_argument('-dn', '--dataset_name', type=str, default='ml-1m', help='Dataset')  # lastfm, citeulike-a, filmtrust, steam
    parser.add_argument('-ln', '--logger_name', type=str, default='', help='Logger name')
    parser.add_argument('-r', '--regular', default=False, help='Is regular training', action='store_true')
    parser.add_argument('--config_path', type=str, default='./config/dataset/', help='Config path')
    parser.add_argument('--saved_path', type=str, default='./saved/', help='Saved path')
    parser.add_argument('--result_path', type=str, default='./result/', help='The path to save the results')
    parser.add_argument('-utb', '--use_tensorboard', default=False, help='Use tensorboard or not', action='store_true')
    parser.add_argument('--tensorboard_path', type=str, default='./log_tensorboard/', help='Path for log of tensorboard')
    
    # model setting
    parser.add_argument('--hidden_dim', type=int, nargs='+', default=[120, 40])
    parser.add_argument('--dropout_prob', type=float, default=0.5, help='The drop out probability of input')
    parser.add_argument('--beta', type=float, default=0.2, help='The default hyperparameter of the weight of KL loss')
    parser.add_argument('--gamma', type=float, default=0.005, help='The hyperparameter shared across all users')
    parser.add_argument('--mixture_weights', type=float, nargs='+', default=[0.15, 0.75, 0.1], help='The mixture weights of three composite priors')
    
    # training setting
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight dacay')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stop', type=int, default=100, help='Patience for early stop')
    parser.add_argument('-alt', '--alternating', default=False, help='Alternative training', action='store_true')
    parser.add_argument('--n_enc_epochs', type=int, default=3, help='The training times of encoder per epoch')
    parser.add_argument('--n_dec_epochs', type=int, default=1, help='The training times of decoder per epoch')
    
    # evaluation setting
    parser.add_argument('--top_k', type=int, default=5, help='Top-k of metrics')
    
    # defense setting
    parser.add_argument('-pm', '--perturb_method', type=str, default='MPC', help='Method to perturb gradient')
    parser.add_argument('--l1_norm_clip', type=float, default=0.1, help='L1 norm clipping value')
    parser.add_argument('--lam', type=float, default=0.05, help='Scale of Laplace distributions')
    parser.add_argument('--xi', type=int, default=3, help='Number of clients to be sent')
    parser.add_argument('--rho', type=int, default=1, help='Ratio of fixed samples')
    parser.add_argument('--tau', type=int, default=0, help='Ratio of random samples')
    parser.add_argument('--enc_module_name', type=str, nargs='+', default=['encoder.fc1.weight'], help='Name of encoder module to protect')
    parser.add_argument('--dec_module_name', type=str, nargs='+', default=['decoder.weight', 'decoder.bias'], help='Name of decoder module to protect')
    
    # attack setting
    parser.add_argument('--restore_epochs', type=int, nargs='*', default=[], help='Epochs to restore user behaviors')
    parser.add_argument('-ueg', '--use_enc_grad', default=False, help='Use gradient of encoder to restore', action='store_true')
    
    return parser.parse_known_args()[0]


if __name__ == '__main__':
    args = parse_args()
    init_seed(args.seed, args.reproducibility)
    
    init_logger(args)
    logger = getLogger()
    log_args(args, logger)
    dataset = load_data(os.path.join(args.config_path, f'{args.dataset_name.lower()}.json'))
    logger.info('\n' + str(dataset))
    
    model_name = args.model_name
    
    if args.regular:
        logger.info('Regular training')
        trainer = Trainer(args, dataset)
    else:
        logger.info('Federated training')
        # construct clients
        clients = Clients(args, dataset)
        # construct the server
        trainer = Server(args, dataset, clients)
    
    trainer.train()
    trainer.test(save = False)
