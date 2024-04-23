import argparse
import yaml
import os
import os.path as osp
import numpy as np

from continuum import ClassIncremental
from continuum.datasets import CIFAR100
from continuum import rehearsal
from torchvision.datasets import CIFAR10 as CIFAR10_torch

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from performance_metrics import identification

from params import ExperimentParams
from utils import create_pairs, update_criterion_weight, BalancedBatchSampler
from model import ResNet32Cifar
from train import train, classification
from eval import validation, evaluate
from datetime import datetime
# load params from the config file from yaml to dataclass
parser = argparse.ArgumentParser(description='CL2R: Compatible Lifelong Learning Represenations')
parser.add_argument("--config_path",
                    help="path of the experiment yaml",
                    default=os.path.join(os.getcwd(), "config.yml"),
                    type=str)
params = parser.parse_args()

with open(params.config_path, 'r') as stream:
    loaded_params = yaml.safe_load(stream)

args = ExperimentParams()
for k, v in loaded_params.items():
    args.__setattr__(k, v)

args.yaml_name = os.path.basename(params.config_path)

args.device = torch.device("cuda")
print(f"Current args:\n{vars(args)}")

# dataset
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1811)
np.random.seed(1811)
data_path = osp.join(args.root_folder, "data")
if not osp.exists(data_path):
    os.makedirs(data_path)
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if not osp.exists(osp.join(args.root_folder, f"checkpoints-{time}")):
    os.makedirs(osp.join(args.root_folder, f"checkpoints-{time}"))
args.checkpoint_path = osp.join(args.root_folder, f"checkpoints-2024-04-21_00-36-58")
    
print(f"Starting Evaluation")
args.nb_tasks = 5
query_set, gallery_set = create_pairs(data_path=data_path)
query_loader = DataLoader(query_set, batch_size=args.batch_size, 
                            shuffle=False, drop_last=False, 
                            num_workers=args.num_workers)
gallery_loader = DataLoader(gallery_set, batch_size=args.batch_size,
                            shuffle=False, drop_last=False, 
                            num_workers=args.num_workers)
evaluate(args, query_loader, gallery_loader)

print(f"args:\n{vars(args)}")