import os
import re
import random
import datetime
import logging
import colorlog
from typing import Union, Iterable
from torch._six import inf
from argparse import Namespace
from logging import Logger

import numpy as np
import torch
from numba import jit

from colorama import init


def get_datetime_str(style=0):
    '''
    style=0返回日期_时间字符串
    style=1返回日期字符串
    style=2返回时间字符串
    '''
    time_now = datetime.datetime.now()
    strData = time_now.strftime('%y%m%d')
    strTime = time_now.strftime('%H%M%S')

    if style == 1:
        return strData
    elif style == 2:
        return strTime
    else:
        return strData + '_' + strTime


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility: 
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class RemoveColorFilter(logging.Filter):

    def filter(self, record):
        if record:
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            record.msg = ansi_escape.sub('', str(record.msg))
        return True


log_colors_config = {
    'DEBUG': 'cyan',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'red',
}


def init_logger(args):
    """
    A logger that can show a message on standard output and write it into the
    file named `filename` simultaneously.
    All the message that you want to log MUST be str.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Example:
        >>> logger = logging.getLogger(args)
        >>> logger.debug(train_state)
        >>> logger.info(train_result)
    """
    init(autoreset=True)
    # LOGROOT = './log/'
    # dir_name = os.path.dirname(LOGROOT)
    # ensure_dir(dir_name)
    # model_name = os.path.join(dir_name, args.model_name)
    # ensure_dir(model_name)
    logfilepath = './log/{}/{}/{}{}-{}.log'.format('regular' if args.regular else 'federated', 
                                                   args.model_name, 
                                                   args.dataset_name,
                                                   f'-{args.logger_name}' if args.logger_name else '', 
                                                   get_datetime_str())
    dir_name = os.path.dirname(logfilepath)
    ensure_dir(dir_name)

    # logfilepath = os.path.join(LOGROOT, logfilename)

    filefmt = "%(asctime)-15s %(levelname)s  %(message)s"
    filedatefmt = "%a %d %b %Y %H:%M:%S"
    fileformatter = logging.Formatter(filefmt, filedatefmt)

    sfmt = "%(log_color)s%(asctime)-15s %(levelname)s  %(message)s"
    sdatefmt = "%d %b %H:%M"
    sformatter = colorlog.ColoredFormatter(sfmt, sdatefmt, log_colors=log_colors_config)
    if args.state is None or args.state.lower() == 'info':
        level = logging.INFO
    elif args.state.lower() == 'debug':
        level = logging.DEBUG
    elif args.state.lower() == 'error':
        level = logging.ERROR
    elif args.state.lower() == 'warning':
        level = logging.WARNING
    elif args.state.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    fh = logging.FileHandler(logfilepath)
    fh.setLevel(level)
    fh.setFormatter(fileformatter)
    remove_color_filter = RemoveColorFilter()
    fh.addFilter(remove_color_filter)

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(sformatter)

    logging.basicConfig(level=level, handlers=[sh, fh], force=True)
    # logging.basicConfig(filename=logfilepath, level=level)

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]
def clip_norm_(parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].device
    if norm_type == inf:
        total_norm = max(p.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in parameters]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.detach().mul_(clip_coef.to(p.device))
    return total_norm


def log_args(args: Namespace, logger: Logger):
    info = "\n[Training Settings]"
    for name, params in args._get_kwargs():
        info += f'\n{name}: {params}'
    logger.info(info)


# @jit(nopython=True)
# def get_sent_items(uids: np.uint64, n_items: np.uint64, clients_data: dict, fixed_items: dict, items_candidate_for_rand: dict, tau: np.uint64):
#     sent_items = np.zeros((len(uids), n_items), dtype=np.uint64)
#     for i, uid in enumerate(uids):
#         i_u = clients_data[uid].nonzero(as_tuple=True)[0].numpy()
#         sent_items[i, i_u] = 1
        
#         # fixed items for protecting interaction behaviors
#         sent_items[i, fixed_items[uid]] = 1
        
#         # sample items for model training
#         if tau > 0:
#             item_candidate = items_candidate_for_rand[uid]
#             neg_num = int(min(tau * i_u.shape[0], item_candidate.shape[0]))
#             neg_items = np.random.choice(item_candidate, neg_num, replace=False)
#             sent_items[i, neg_items] = 1
    

@jit(nopython=True)
def sample_neighbor(n_neighbors, n_samples):
    neighbor_sample = np.empty((n_neighbors, n_samples))
    for i in range(n_neighbors):
        neighbor_sample[i] = np.random.choice(n_neighbors, n_samples, False)
    bias = (np.arange(n_neighbors) + 1).reshape(n_neighbors, -1)
    neighbor_sample = (neighbor_sample + bias) % n_neighbors
    return neighbor_sample.astype(np.int_)


@jit(nopython=True)
def get_upload_items(sent_items, sent_neighbors_idx):
    upload_items = np.zeros_like(sent_items)
    for i, idxs in enumerate(sent_neighbors_idx):
        # u
        upload_items[i] |= sent_items[i]
        # u's neighbors
        for idx in idxs:
            upload_items[i] |= sent_items[idx]
    return upload_items


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    if type(obj) is torch.Tensor:
        return obj.numel() * obj.element_size()
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    else:
        if hasattr(obj, '__dict__'):
            size += get_size(obj.__dict__, seen)
        if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            size += sum([get_size(i, seen) for i in obj])
    return size


if __name__ == '__main__':
    strData=get_datetime_str(style=1)
    print(strData)#220901,  2022年9月1日
    strData=get_datetime_str(2)
    print(strData)#191018， 19:10:18
    strData=get_datetime_str()
    print(strData)

    print('{0}.py was saved'.format(strData))
    '''
    220901
    191018
    220901_191018
    220901_191018.py was saved
    '''
