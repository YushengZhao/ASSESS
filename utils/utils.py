import torch
import random
import numpy as np
import logging
import time
import os
import networkx as nx


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def create_mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


def get_logger(args):
    create_mkdir(args.log_dir)
    log_path = os.path.join(args.log_dir, args.DS+'_'+args.log_file)
    print('logging into %s' % log_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info('#' * 20)

    # record arguments
    args_str = ""
    for k, v in sorted(vars(args).items()):
        args_str += "%s" % k + "=" + "%s" % v + "; "
    logger.info(args_str)
    print(args_str)
    logger.info("DS: %s" % args.DS)
    logger.info(f'Split: {args.data_split}, Source Index: {args.source_index}, Target Index: {args.target_index}')

    return logger

def neighborhood(G, node, n):
    paths = nx.single_source_shortest_path(G, node)
    return [node for node, traversed_nodes in paths.items()
            if len(traversed_nodes) == n+1]

def save_model(ckpt_dir, model):
    saved_state = {
        'model': model.state_dict(),
    }
    torch.save(saved_state, ckpt_dir)

def load_model(ckpt_dir, model):
    saved_state = torch.load(ckpt_dir)
    model.load_state_dict(saved_state['model'])
    return model
