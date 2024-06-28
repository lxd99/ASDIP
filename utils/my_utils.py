import logging
import dgl
import torch
import numpy
import random
import numba
import argparse


def get_logger(log_path):
    logging.getLogger('matplotlib.font_manager').disabled = True
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'{log_path}', mode='w')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def convert_list_to_label(labels, user_num, device):
    label_indices = [[row, col] for row, x in enumerate(labels) for col in x]
    label_tensor = torch.sparse_coo_tensor(torch.tensor(label_indices).t(), [1] * len(label_indices),
                                           (len(labels), user_num), device=device).to_dense()
    return label_tensor


def setup_seed(seed, torch_only_deterministic=True):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    # dgl.seed(seed)
    # cuda pick the same algorithm rather than select the fastest
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # the picked algorithm of cuda is deterministic
    if torch_only_deterministic:
        torch.use_deterministic_algorithms(True)  # the algorithm picked by torch is deterministic


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def merge_args(args, dict_args):
    opt = vars(args)
    for key in dict_args:
        if opt[key] is None:
            opt[key] = dict_args[key]
    return argparse.Namespace(**opt)


def setup_thread(thread=5):
    torch.set_num_threads(thread)
    numba.set_num_threads(thread)
