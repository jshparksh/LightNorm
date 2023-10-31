
from utils.logger import Log

import torch

# Load models
from model.AlexNet import AlexNetCifar
from model.ResNet import ResNet18Cifar, ResNet50Cifar
from model.DenseNet import DenseNet121Cifar
from model.MobileNetv1 import MobileNetv1Cifar
from model.MobileNetv2 import MobileNetV2ImageNet, MobileNetv2Cifar
from model.VGG import VGG16Cifar

from model.MLPMixer import mlp_mixer_b16

import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

from bfp.functions import ReplaceLayers

<<<<<<< HEAD
def GetNetwork(dataset, model, num_classes = 10, bfp_conf = None, dtype = "fp32", pretrained = False):
=======
def GetNetwork(dataset, model, num_classes = 10, bfp_conf = None, pretrained = False, silence = False):
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
    if dataset.lower() == "imagenet":
        if model.lower() == 'mobilenetv2':
            net = MobileNetV2ImageNet(num_classes)
        if model.lower() == 'mobilenetv1':
            net = MobileNetV2ImageNet(num_classes)
        if model.lower() in model_names:
            if pretrained:
                net = models.__dict__[model](pretrained=True)
                if not silence:
                    Log.Print("Using pretrained pytorch {model} imagenet model...", current=False, elapsed=False)
            else:
                net = models.__dict__[model]()
                if not silence:
                    Log.Print("Using pytorch {model} imagenet model...", current=False, elapsed=False)
        else:
            NotImplementedError("Imagenet model {model} not defined on pytorch")
    elif dataset.lower() in ["cifar10", "cifar100"]:
        if model.lower() == "alexnet":
            net = AlexNetCifar(num_classes)
        elif model.lower() == "resnet18":
            net = ResNet18Cifar(num_classes)
        elif model.lower() == "resnet50":
            net = ResNet50Cifar(num_classes)
        elif model.lower() == "densenet121":
            net = DenseNet121Cifar(num_classes)
        elif model.lower() == "mobilenetv1":
            net = MobileNetv1Cifar(num_classes)
        elif model.lower() == 'mobilenetv2':
            net = MobileNetv2Cifar(num_classes)
        elif model.lower() == "vgg16":
            net = VGG16Cifar(num_classes)
        else:
            NotImplementedError("Model {model} on cifar10/cifar100 not implemented on ./models/ folder.")
    else:
        NotImplementedError("Dataset {datset} not supported.")

    if bfp_conf != None:
<<<<<<< HEAD
        ReplaceLayers(net, bfp_conf, dtype)
        Log.Print("Replacing model's layers to provided bfp config...", current=False, elapsed=False)
=======
        if not silence:
            Log.Print("Replacing model's layers to provided bfp config...", current=False, elapsed=False)
        ReplaceLayers(net, bfp_conf, silence = silence)
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
    return net


import torch.optim as optim


def GetOptimizer(args, epoch, silence=False):
    if str(epoch) in args.optimizer_dict:
        if not silence:
            Log.Print("Setting optimizer from dict", elapsed=False, current=False)
        config = args.optimizer_dict[str(epoch)]
        lr = config["lr-initial"] if "lr-initial" in config else args.optim_lr
        momentum = config["momentum"] if "momentum" in config else args.optim_momentum
        weight_decay = config["weight-decay"] if "weight-decay" in config else args.optim_weight_decay
    else:
        if not silence:
            Log.Print("Configuration not found. Returning default Optimizer...", elapsed=False, current=False)
        lr = args.optim_lr
        momentum = args.optim_momentum
        weight_decay = args.optim_weight_decay

    opt = optim.SGD(args.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    if str(epoch) in args.optimizer_dict:
        if "step" in config:
            for i in range(config["step"]):
                opt.step()

    return opt

# Scheduler also uses same dict with optimizer
<<<<<<< HEAD
def GetScheduler(args, epoch):
    MILESTONES = [60, 120, 160]
=======
def GetScheduler(args, epoch, silence=False):
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
    if str(epoch) in args.optimizer_dict:
        if not silence:
            Log.Print("Setting scheduler from dict", elapsed=False, current=False)
        config = args.optimizer_dict[str(epoch)]
    else:
<<<<<<< HEAD
        Log.Print("Configuration not found. Returning default Scheduler from args...", elapsed=False, current=False)
    
    #sche = optim.lr_scheduler.MultiStepLR(args.optimizer, milestones=MILESTONES, gamma=0.2)
=======
        if not silence:
            Log.Print("Configuration not found. Returning default Scheduler from args...", elapsed=False, current=False)
        
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
    sche = optim.lr_scheduler.CosineAnnealingLR(args.optimizer, T_max=args.training_epochs)
        
    if str(epoch) in args.optimizer_dict:
        config = args.optimizer_dict[str(epoch)]
        if "step" in config:
            for i in range(config["step"]):
                sche.step()

    return sche


def GetDefOptimizer(args, epoch):
    lr = args.optim_lr
    momentum = args.optim_momentum
    weight_decay = args.optim_weight_decay

    opt = optim.SGD(args.net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for i in range(epoch):
        opt.step()

    return opt


# Scheduler also uses same dict with optimizer
def GetDefScheduler(args, epoch):
    sche = optim.lr_scheduler.CosineAnnealingLR(args.optimizer, T_max=args.training_epochs)
        
    for i in range(epoch):
        sche.step()

    return sche