import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torch.utils.data.sampler import BatchSampler

import numpy as np
from PIL import Image
from collections import defaultdict as dd
from torchvision.datasets import CIFAR10 as CIFAR10_torch

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class ImagesDataset(Dataset):
    def __init__(self, data=None, targets=None, transform=None):
        self.data = data
        self.targets = targets
        self.transform = None if transform is None else transforms.Compose(transform)

    def __getitem__(self, index):
        if isinstance(self.data[index], str):
            x = Image.open(self.data[index]).convert('RGB')
        else:
            if self.transform: 
                x = Image.fromarray(self.data[index].astype(np.uint8))
            else:
                x = self.data[index]

        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, n_classes, n_samples, seen_classes, rehearsal=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.seen_classes = seen_classes
        self.rehearsal = rehearsal
        self.n_batches = self.n_samples // self.batch_size # drop last
        if self.n_batches == 0:
            self.n_batches = 1
            self.size = self.n_samples if rehearsal == 0 else self.n_samples//2
        elif rehearsal == 0:
            self.size = self.batch_size
        else:
            self.size = self.batch_size//2
        self.index_dic = dd(list)
        self.indices = []
        self.seen_indices = []
        for index, y in enumerate(self.dataset._y):
            if y not in self.seen_classes:
                self.indices.append(index)
            else:
                self.seen_indices.append(index)

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            if self.rehearsal > 0:
                replace = True if len(self.seen_indices) <= self.size else False
                batch.extend(np.random.choice(self.seen_indices, size=self.size, replace=replace))
            replace = True if len(self.indices) <= self.size else False
            batch.extend(np.random.choice(self.indices, size=self.size, replace=replace))
            yield batch

    def __len__(self):
        return self.n_batches


def create_pairs(data_path, num_pos_pairs=3000, num_neg_pairs=3000):

    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                        (0.2675, 0.2565, 0.2761))
                                    ])

    gallery_set = CIFAR10_torch(root=data_path, 
                                train=False, 
                                download=True, 
                                transform=transform
                               )
    
    query_set = CIFAR10_torch(root=data_path, 
                              train=True, 
                              download=True, 
                              transform=transform
                             )    
    return query_set, gallery_set
   

def update_criterion_weight(args, num_old_classes, num_new_classes):
    if args.type_update_criterion_weight == "cl2r":
        args.criterion_weight = args.criterion_weight_base * np.sqrt(num_new_classes / num_old_classes)
    elif args.type_update_criterion_weight == "pod":
        args.criterion_weight = args.criterion_weight_base * np.sqrt(num_old_classes / num_new_classes)
    return args.criterion_weight

def update_spatial_lambda_c(args, num_old_classes, num_new_classes):
    if args.type_update_criterion_weight == "cl2r":
        args.spatial_lambda_c = args.spatial_lambda_c_base * np.sqrt(num_new_classes / num_old_classes)
    elif args.type_update_criterion_weight == "pod":
        args.spatial_lambda_c = args.spatial_lambda_c_base * np.sqrt(num_old_classes / num_new_classes)
    return args.spatial_lambda_c

  
def extract_features(args, net, loader, return_labels=False):
    features = None
    labels = None
    net.eval()
    with torch.no_grad():
        for inputs in loader:
            images = inputs[0].to(args.device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.amp):
                f = net(images)['features']
            f = l2_norm(f)
            if features is not None:
                features = torch.cat((features, f), 0)
                labels = torch.cat((labels, inputs[1]), 0) if return_labels else None
            else:
                features = f
                labels = inputs[1] if return_labels else None
    if return_labels:
        return features.detach().cpu(), labels.detach().cpu()
    return features.detach().cpu().numpy()


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_epoch(n_epochs=None, loss=None, acc=None, epoch=None, task=None, time=None, classification=False):
    acc_str = f"Task {task + 1}" if task is not None else f""
    acc_str += f" Epoch [{epoch + 1}]/[{n_epochs}]" if epoch is not None else f""
    acc_str += f"\t Training Loss: {loss:.4f}" if loss is not None else f""
    acc_str += f"\t Training Accuracy: {acc:.2f}" if acc is not None else f""
    acc_str += f"\t Time: {time:.2f}" if time is not None else f""
    if classification:
        acc_str = acc_str.replace("Training", "Classification")   
    print(acc_str)
