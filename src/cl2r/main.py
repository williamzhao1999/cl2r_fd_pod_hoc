import argparse
import yaml
import os
import os.path as osp
import numpy as np
import random

from continuum import ClassIncremental
from continuum.datasets import CIFAR100
from continuum import rehearsal

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from cl2r.params import ExperimentParams
from cl2r.utils import create_pairs, update_criterion_weight, BalancedBatchSampler, update_spatial_lambda_c
from cl2r.model import ResNet32Cifar
from cl2r.train import train, classification
from cl2r.eval import validation, evaluate
from datetime import datetime

from cl2r.hoc import HocLoss

def main():
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

    args.device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    print(f"Current args:\n{vars(args)}")
    
    # dataset
    # reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1811)
    np.random.seed(1811)
    random.seed(1811)
    data_path = osp.join(args.root_folder, "data")
    if not osp.exists(data_path):
        os.makedirs(data_path)
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not osp.exists(osp.join(args.root_folder, f"checkpoints-{time}")):
        os.makedirs(osp.join(args.root_folder, f"checkpoints-{time}"))
    args.checkpoint_path = osp.join(args.root_folder, f"checkpoints-{time}")
        
    print(f"Loading Training Dataset")
    train_transform = [transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5071, 0.4867, 0.4408),
                                            (0.2675, 0.2565, 0.2761))
                       ]
    dataset_train = CIFAR100(data_path=data_path, train=True, download=True)
    # create task-sets for lifelong learning
    scenario_train = ClassIncremental(dataset_train,
                                      initial_increment=args.start,
                                      increment=args.increment,
                                      transformations=train_transform)

    args.num_classes = scenario_train.nb_classes
    args.nb_tasks = scenario_train.nb_tasks

    val_transform = [transforms.ToTensor(),
                       transforms.Normalize((0.5071, 0.4867, 0.4408),
                                            (0.2675, 0.2565, 0.2761))
                    ]
    dataset_val = CIFAR100(data_path=data_path, train=False, download=True)
    # create task-sets for lifelong learning
    scenario_val = ClassIncremental(dataset_val,
                                    initial_increment=args.start,
                                    increment=args.increment,
                                    transformations=val_transform)
    
    # create episodic memory dataset
    memory = rehearsal.RehearsalMemory(memory_size=args.num_classes * args.rehearsal,
                                       herding_method="random",
                                       fixed_memory=True,
                                       nb_total_classes=args.num_classes
                                    )


    add_loss = None
    scaler = None
    if args.method == "hoc" or args.method == "hoc_new":
        add_loss = HocLoss(mu_=10)

    print(f"Starting Training")
    for task_id, (train_task_set, _) in enumerate(zip(scenario_train, scenario_val)):
        print(f"Task {task_id+1} Classes in task: {train_task_set.get_classes()}")

        rp = ckpt_path if (task_id > 0) else None
        net = ResNet32Cifar(resume_path=rp, 
                            starting_classes=100, 
                            feat_size=99, 
                            device=args.device,
                            args=args)

        if task_id > 0:
            previous_net = ResNet32Cifar(resume_path=ckpt_path, 
                                         starting_classes=100, 
                                         feat_size=99, 
                                         device=args.device,
                                         args=args)
            previous_net.eval() 
        else:
            previous_net = None
        
        print(f"Created model {'and old model' if task_id > 0 else ''}")

        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-04)
        scheduler_lr = MultiStepLR(optimizer, milestones=args.stages, gamma=0.1)
        criterion_cls = nn.CrossEntropyLoss().cuda(args.device)

        if task_id > 0:
            if args.fixed_weight == True:
                args.criterion_weight = args.criterion_weight_base
                if args.pod_loss == True:
                    args.spatial_lambda_c = args.spatial_lambda_c_base
            else:
                args.criterion_weight = update_criterion_weight(args, args.seen_classes.shape[0], train_task_set.nb_classes)
                if args.pod_loss == True:
                    args.spatial_lambda_c = update_spatial_lambda_c(args, args.seen_classes.shape[0], train_task_set.nb_classes)

            # print("args.seen_classes.shape[0]", args.seen_classes.shape[0], "train_task_set.nb_classes", train_task_set.nb_classes
            #, "args.criterion_weight",args.criterion_weight,"args.spatial_lambda_c", args.spatial_lambda_c)
            mem_x, mem_y, mem_t = memory.get()
            train_task_set.add_samples(mem_x, mem_y, mem_t)
            batchsampler = BalancedBatchSampler(train_task_set, n_classes=train_task_set.nb_classes, 
                                                    batch_size=args.batch_size, n_samples=len(train_task_set._x), 
                                                    seen_classes=args.seen_classes, rehearsal=args.rehearsal)
            train_loader = DataLoader(train_task_set, batch_sampler=batchsampler, num_workers=args.num_workers) 
        else:
            train_loader = DataLoader(train_task_set, batch_size=args.batch_size, shuffle=True, 
                                      drop_last=True, num_workers=args.num_workers) 

        val_dataset = scenario_val[:task_id + 1]
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                sampler=None, drop_last=False, num_workers=args.num_workers)
        
        best_acc = 0
        print(f"Starting Epoch Loop at task {task_id + 1}/{scenario_train.nb_tasks}")
        for epoch in range(args.epochs):
            train(args, net, train_loader, optimizer, epoch, criterion_cls, previous_net, task_id, add_loss, scaler)
            # acc_val = validation(args, net, query_loader, gallery_loader, task_id, selftest=(task_id == 0))
            # if task_id > 0:
            #     acc_val = validation(args, query_loader, gallery_loader, task_id)
            # else:
            scheduler_lr.step()
            acc_val = classification(args, net, val_loader, criterion_cls)
            
            if (acc_val > best_acc and args.save_best) or (not args.save_best and epoch == args.epochs - 1):
                best_acc = acc_val
                print("Saving model")
                ckpt_path = osp.join(*(args.checkpoint_path, f"ckpt_{task_id}.pt"))
                torch.save(net.state_dict(), ckpt_path)
        
        memory.add(*scenario_train[task_id].get_raw_samples(), z=None)  
        args.seen_classes = torch.tensor(list(memory.seen_classes), device=args.device)
        
    print(f"Starting Evaluation")
    
    query_set, gallery_set = create_pairs(data_path=data_path)
    query_loader = DataLoader(query_set, batch_size=args.batch_size, 
                              shuffle=False, drop_last=False, 
                              num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_set, batch_size=args.batch_size,
                                shuffle=False, drop_last=False, 
                                num_workers=args.num_workers)
    evaluate(args, query_loader, gallery_loader)

    print(f"args:\n{vars(args)}")


if __name__ == '__main__':
    main()