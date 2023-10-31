import torch
import torch.optim as optim
import torch.nn as nn



from utils.logger import Log
from utils.slackBot import slackBot
from utils.generateConfig import GenerateConfig
from utils.tensorAnalyze import TensorAnalyze
from utils.statManager import statManager
from utils.functions import str2bool

from train.dataset import LoadDataset
from train.network import GetNetwork, GetOptimizer, GetScheduler
from train.train import TrainNetwork
from bfp.functions import LoadBFPDictFromFile
from torchsummary import summary
#from torch.utils.tensorboard import SummaryWriter

import os
import json
import argparse
from datetime import datetime
import wandb

args = None


def SetArgsFromConf(args, attr_name):
    tc = getattr(args, "train_config")
    if tc != None and attr_name in tc:
        setattr(args, attr_name.replace("-","_"), tc[attr_name])

# Parse arguments
def ArgumentParse():
    parser = argparse.ArgumentParser()

    # Global options
    parser.add_argument("--mode", type=str, default="train",
        help = "Program execution mode [train, analyze, generate-config]")
    parser.add_argument("--cuda", type=str2bool, default=True,
        help = "True if you want to use cuda")

    # tag:trainConfig
    parser.add_argument("-tc", "--train-config-file", type=str, default="",
        help = "Train config file. Please see documentation about usage")

    # ----------------- configurable by train config file -------------------
    # tag:Save
    parser.add_argument("--save-name", type=str, default = str(datetime.now())[:-7].replace("-","").replace(":","").replace(" ","_"),
        help = "Name of the saved log file, stat object, save checkpoint")
    parser.add_argument("--pickle-path", type=str, default = "./_pickles",
        help = "Set location to save pickle file")
    parser.add_argument("--log", type=str2bool, default = True,
        help = "Set true to save log file")
    """parser.add_argument("--slackbot", type=str2bool, default = True,
        help = "Set true to send message to slackbot")"""
    parser.add_argument("--stat", type=str2bool, default = False,
        help = "Record to stat object?")
    parser.add_argument("--save", type=str2bool, default = False,
        help = "Set true to save checkpoints")
    parser.add_argument("--save-interval", type=int, default = 0,
        help = "Checkpoint save interval. 0:only last, rest:interval")
    parser.add_argument("--dtype", type=str, default="FP32",
        help = "Data precision to uss [FP32, FP16, FP16+4, FP16+8, FP8, FP8+4, FP8+8, BFLOAT16")
    
    
    # tag:Dataset
    parser.add_argument("-d","--dataset", type=str, default = "CIFAR100",
        help = "Dataset to use [CIFAR10, CIFAR100, ImageNet]")
    parser.add_argument("-dp","--dataset-path", type=str, default = "./data",
        help = "Provide if dataset is prepared, escpecially for ImageNet. Dataset will automatically downloaded on CIFAR-10 and CIFAR-100")
    parser.add_argument("--dataset-pin-memory", type=str2bool, default = True,
        help = "Setting this option will pin dataset to memory, which may boost execution speed.")
    parser.add_argument("--num-workers", type=int, default = 0,
        help = "Recommended value : 4")
    parser.add_argument("--batch-size-train", type=int, default = 256, 
        help = ".") #128
    parser.add_argument("--batch-size-test", type=int, default = 100,
        help = ".")

    # tag:Model
    parser.add_argument("-m","--model", type=str, default = "ResNet18",
        help = "Model [AlexNet, ResNet18, MobileNetv1, DenseNetCifar, VGG16, MLPMixerB16]")
    parser.add_argument("-bfp", "--bfp-layer-conf-file", type=str, default="",
        help = "Config of the bfp setup, if not set, original network will be trained")
    
    # tag:Train
    parser.add_argument("--training-epochs", type=int, default = 200,
        help = ".")
    parser.add_argument("--start-epoch", type=int, default = 0,
        help = ".")
    parser.add_argument("--optim-lr", type=float, default = 0.01,
        help = "Optimizer learning rate")
    parser.add_argument("--optim-momentum", type=float, default = 0.9,
        help = "Optimizer momentum")
    parser.add_argument("--optim-weight-decay", type=float, default = 5e-4,
        help = "Optimizer weight decay")
    parser.add_argument("--loss-boost", type=float, default = 1.0,
        help = "Loss boost to each layer [NOT IMPLEMENTED]")

    # tag:Print
    parser.add_argument("--print-train-batch", type=int, default = 0,
        help = "Print progress on # of batches, 0 to disable") # 128 = 391
    parser.add_argument("--print-train-count", type=int, default = 5,
        help = "# of print progress on each epoch, 0 to disable") 

    # ------------- end of configurable by train config file ---------------

    # Tag:zseAnalyze
    parser.add_argument("--save-file", type=str, default = "",
        help = "Saved checkpoint of the model")
    parser.add_argument("--zse-bfp", type=str2bool, default = False,
        help = "[zse-analyze] If saved file is BFP network, set this to true")
    parser.add_argument("--zse-graph-mode", type=str, default="percentage",
        help = "[zse-analyze] Choose graph mode [none, percentage, count]")
    parser.add_argument("--zse-print-mode", type=str, default = "sum",
        help = "[zse-analyze] Choose print mode [none, sum, format, all]")

    # Parse arguments
    args = parser.parse_args()

    # tag:trainConfig
    if args.train_config_file == "":
        Log.Print("Train Config file not set. Using command line options...", current=False, elapsed=False)
        args.train_config = None
    elif not os.path.exists("./conf_train/"+args.train_config_file+".json"):
        raise FileNotFoundError(args.train_config_file + ".json not exists on ./conf_train/ directory!")
    else:
        with open("./conf_train/"+args.train_config_file+".json", "r", encoding="utf-8") as f:
            args.train_config = json.load(f)

    # tag:Save
    SetArgsFromConf(args, "save-name")
    SetArgsFromConf(args, "log")
    SetArgsFromConf(args, "stat")
    SetArgsFromConf(args, "save")
    """SetArgsFromConf(args, "slackbot")"""
    SetArgsFromConf(args, "save-interval")

    # Additional handlers
    if not os.path.exists("./_logs"):
        os.makedirs("./_logs")
    if not os.path.exists("./_saves"):
        os.makedirs("./_saves")
    if not os.path.exists("./_stats"):
        os.makedirs("./_stats")
    args.log_location = "./_logs/" + args.save_name + ".log"
    args.save_prefix = "./_saves/" + args.save_name
    args.stat_location = "./_stats/" + args.save_name + ".stat"
    # Set the log file
    Log.SetLogFile(True, args.log_location) if args.log else Log.SetLogFile(False)
    # args.stat = statManager() if args.stat else None
    """slackBot.Enable() if args.slackbot else slackBot.Disable()
    slackBot.SetProcessInfo(args.save_name)"""
    
    # tag:Dataset
    SetArgsFromConf(args, "dataset")
    SetArgsFromConf(args, "dataset-path")
    SetArgsFromConf(args, "batch-size-train")
    SetArgsFromConf(args, "batch-size-test")

    SetArgsFromConf(args, "dataset-pin-memory")
    SetArgsFromConf(args, "num-workers")
    
    # tag:Train
    SetArgsFromConf(args, "training-epochs")
    SetArgsFromConf(args, "loss-boost")
    SetArgsFromConf(args, "start-epoch")
    
    # Load dataset
    args.trainset, args.testset, args.classes, args.trainloader, args.testloader = LoadDataset(args)
    args.batch_count = len(args.trainloader)
    # Image Classification
    args.num_classes = len(args.classes)
    
    # tag:Model
    SetArgsFromConf(args, "model")
    SetArgsFromConf(args, "bfp-layer-conf-file")
    args.bfp_layer_conf_dict = dict()
    SetArgsFromConf(args, "bfp-layer-conf-dict")
    if args.bfp_layer_conf_file != "":
        Log.Print("bfp-layer-conf-file is set.", elapsed=False, current=False)
        args.net = GetNetwork(args.dataset, args.model, args.num_classes, LoadBFPDictFromFile(args.bfp_layer_conf_file), args.dtype)
    elif str(args.start_epoch) not in args.bfp_layer_conf_dict:
        Log.Print('bfp-layer-conf-file or bfp-layer-conf-dict is not set. Or, "{args.start_epoch}" is not provided on bfp-layer-conf-dict. Naive network will trained.', elapsed=False, current=False)
        args.net = GetNetwork(args.dataset, args.model, args.num_classes, dict(), args.dtype)
    else:
        Log.Print("bfp-layer-conf-dict is set.", elapsed=False, current=False)
        Log.Print(str(args.bfp_layer_conf_dict))
        args.net = GetNetwork(args.dataset, args.model, args.num_classes, LoadBFPDictFromFile(args.bfp_layer_conf_dict[str(args.start_epoch)]), args.dtype)


    # Critertion, optimizer, scheduler
    args.criterion = nn.CrossEntropyLoss()
    args.optimizer_dict = dict()
    #args.writer = SummaryWriter('runs/'+args.run_dir) 
    SetArgsFromConf(args, "optimizer-dict")
    
    # Optimizer and scheduler
    SetArgsFromConf(args, "optim-lr")
    SetArgsFromConf(args, "optim-momentum")
    SetArgsFromConf(args, "optim-weight-decay")
    args.optimizer = GetOptimizer(args, args.start_epoch)
    args.scheduler = GetScheduler(args, args.start_epoch)
    
    # tag:Print
    SetArgsFromConf(args, "print-train-batch")
    SetArgsFromConf(args, "print-train-count")
    
    if args.cuda: # btw, if device not cuda, I never sure code will work or not
        args.net.to('cuda')
    # TODO : Support Distributed DataParallel

    # Write Tensorboard model
    #dataiter = iter(args.trainloader)
    #images, _ = dataiter.next()
    #args.writer.add_graph(args.net, images.cuda())
    #args.writer.close()
    
    return args

if __name__ == '__main__':
    # Parse Arguments and prepare almost everything
    args = ArgumentParse()
    #summary(args.net, (3, 32, 32))
    Log.Print("Program executed on {} mode.".format(args.mode), current=False, elapsed=False)
    if args.mode == "train":
        # Network training mode
        for arg in vars(args):
            if arg in ["bfp_layer_confs", "bfp_layer_conf", "checkpoints" "trainset", "testset", "classes", "trainloader", "testloader"] or "zse" in arg:
                continue
            Log.Print(str(arg) + " : " + str(getattr(args, arg)), current=False, elapsed=False)
        # Setup Slackbot
        """text_file = open("./slackbot.token", "r")
        data = text_file.read()
        text_file.close()
        slackBot.SetToken(data)
        slackBot.SendStartSignal()"""
        try:
            # Train the Network

            TrainNetwork(args)
        except KeyboardInterrupt:
            Log.Print("Quit from User Signal")
            if args.stat:
                statManager.SaveToFile(args.stat_location)
            if args.save:
                SaveState(suffix = "canceled")
            """if args.slackbot:
                slackBot.SendError("User Interrupted")"""
        # End the Training Signal
        """slackBot.SendEndSignal()"""
    elif args.mode == "analyze":
        for arg in vars(args):
            if arg in ["bfp_layer_confs", "checkpoints" "trainset", "testset", "classes", "trainloader", "testloader", "bfp_layer_conf",
            "criterion", "optimizer", "scheduler", "stat_location", "save_prefix", "loss_boost", "training_epochs", "train_config_file"]:
                continue
            Log.Print(str(arg) + " : " + str(getattr(args, arg)), current=False, elapsed=False)
        # zse analyze mode
        TensorAnalyze(args)
    elif args.mode == "generate-config":
        # Generating config mode
        GenerateConfig(args)
    else:
        raise NotImplementedError("Mode not supported : {}".format(args.mode))

    Log.Print("Program Terminated")