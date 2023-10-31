import torch
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
from utils.logger import Log

def str2tuple(v):
    r = []
    v = v.replace(" ","").replace("(","").replace(")","").split(",")
    for i in v:
        r.append(int(i))
    return tuple(r)

def str2bool(v):
    if v.lower() in ["true", "t", "1"]: return True
    elif v.lower() in ["false", "f", "0"]: return False
    else: raise ValueError("str2bool: not parsable")

def DictKey(d, v):
    for key in d:
        if d[key] == v:
            return key


# Flatten all childrens to 1-d array
def flatten(el):
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res

# Better version of GetAttr, it supports list / sequential too
def getattr_(obj, name):
    name = name.split(".")
    for i in name:
        if i.isdigit():
            obj = obj[int(i)]
        else:
            obj = getattr(obj, i)
    return obj

# Better version of setattr, it supports list / sequential too
def setattr_(obj, name, target):
    name = name.split(".")
    if len(name) > 1:
        for i in name[:-1]:
            if i.isdigit():
                obj = obj[int(i)]
            else:
                obj = getattr(obj, i)
    setattr(obj, name[-1], target)

from torch.optim.lr_scheduler import _LRScheduler

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]