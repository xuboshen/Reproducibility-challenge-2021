import argparse
import builtins
import math
import os
import random
import sys
import shutil
import time
import os.path as osp
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils.tools import mkdir_if_missing
from trainer import *

# import moco.builder

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# 新方法插入到哪里：就trainer那里，其他没了

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--root', type=str, default='../data', help='path to dataset')
parser.add_argument('--output_dir', type = str, default='output/mixstyle/sketch',help='output directory')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate. ')
parser.add_argument('--test_split', default='all', type=str)
parser.add_argument('--final_model', default='best_val', type=str)
parser.add_argument('--backbone', type=str, default='resnet18', help='name of CNN backbone')
parser.add_argument('--source_domains', type=str, default = "", nargs='+', help='source domains for DG')
parser.add_argument('--dataset', default='pacs', type=str)
parser.add_argument('--trainsampler', default='RandomClassSampler', type=str)
parser.add_argument('--n_domains', default=4, type=int)
parser.add_argument('--testsampler', default='RandomClassSampler', type=str)
parser.add_argument('--batch_size', default=126, type=int)
parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--lr_scheduler', default='cosine', type=str)
parser.add_argument('--max_epoch', default=80, type=int)
parser.add_argument('--target_domains', type=str, nargs='+', default="sketch", help='target domains for DG')
parser.add_argument('--trainer', type=str, default='vanilla', help='name of trainer')
parser.add_argument('--n_ins', default=18, type=int) # 每一类采样个数
parser.add_argument('--alpha', default=0.1, type=float)

parser.add_argument('--verbose', default=True, type=bool)
parser.add_argument('--eval_only', action='store_true')
parser.add_argument('--load_epoch', type=int)
parser.add_argument('--model_dir', type=str, default='')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--no_train', action='store_true')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma_for_lr', default=0.1, type=float)
parser.add_argument('--warmup_epoch', default=-1, type=int)
parser.add_argument('--USE_CUDA', default=True, type=bool)
parser.add_argument('--init_weights', default='', type=str)
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--no_test', default=False, type=str)
parser.add_argument('--gpu', default=True, type=bool, help='use_cuda ')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
parser.add_argument('--checkpoint_freq', default=0, type=str)
parser.add_argument('--adam_beta1', default=0.5, type=str)

def main():
    args = parser.parse_args() # 加载参数
    print('Setting fixed seed:{}'.format(args.seed))
    if args.seed is not None: # 设置种子
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    setup_logger(args.output_dir) # 将命令行输出重定向到output文件中
    if args.gpu is not None: # 设置使用gpu
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        torch.backends.cudnn.benchmark = True
    print_args(args)
    trainer = Mytrainer(args) # 实例化一个巨无霸聚合的trainer
    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return
    if not args.no_train:
        trainer.train() # 开始训练

class Logger:
    """Write console output to external text file.
    
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_
    
    Args:
        fpath (str): directory to save logging file.
    
    Examples::
       >>> import sys
       >>> import os.path as osp
       >>> save_dir = 'output/experiment-1'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.file.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def setup_logger(output=None):
    if output is None:
        return

    if output.endswith('.txt') or output.endswith('.log'):
        fpath = output
    else:
        fpath = osp.join(output, 'log.txt')

    if osp.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime('-%Y-%m-%d-%H-%M-%S')

    sys.stdout = Logger(fpath)

def print_args(args):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    
if __name__ == '__main__':
    main()
