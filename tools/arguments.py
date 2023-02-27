import argparse
import os
import torch

import numpy as np
import random

import re 
import yaml

import warnings

from datetime import datetime


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                self.__dict__[key] = value
    
    def __getattr__(self, attribute):

        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=8)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--data_dir', type=str, default=os.getenv('DATA'))
    parser.add_argument('--log_dir', type=str, default=os.getenv('LOG'))
    parser.add_argument('--ckpt_dir', type=str, default=os.getenv('CHECKPOINT'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--save_feature_only', action='store_true', help='save encoded feature only in linear_eval.py')
    parser.add_argument('--percent', type=float, default=1.0)
    parser.add_argument('--start_epoch', type=int, default=0, help="continue training from certain epoch")
    parser.add_argument('--end_epoch', type=int, default=800, help="end training at certain epoch")
    parser.add_argument('--save_interval', type=int, default=100, help="save every x epochs")
    parser.add_argument('--save_dir', type=str, default=None, help="should be specified when start_epoch>0")
    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--linear', action='store_true', help='linear eval in eval_model.py')
    parser.add_argument('--output', type=str, default=None, help='output file for eval_model.py')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    if args.debug:
        if args.train: 
            args.train.batch_size = 2
            args.train.num_epochs = 1
            args.train.stop_at_epoch = 1
        if args.eval: 
            args.eval.batch_size = 2
            args.eval.num_epochs = 1 # train only one epoch
        args.dataset.num_workers = 0


    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]

    set_deterministic(args.seed)

    vars(args)['aug_kwargs'] = {
        'name':args.model.name,
        'mean_std_name': args.model.mean_std_name,
        'image_size': args.dataset.image_size
    }
    vars(args)['dataset_kwargs'] = {
        'dataset':args.dataset.name,
        'data_dir': args.data_dir,
        'download':args.download,
        'max_dataset_size': args.dataset.max_dataset_size,
        'num_imagenet_classes': args.dataset.num_imagenet_classes, 
        'imagenet_size': args.dataset.imagenet_size, 
        'debug_subset_size': args.debug_subset_size if args.debug else None,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    return args
