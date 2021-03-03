import torch
import torch.optim as optim
import torch.nn as nn

from net import SimpleNet, ResNet18, BFSimpleNet, BFResNet18
from log import Log
from functions import LoadDataset

import signal
import sys
import argparse
import math

def str2bool(v):
    if v.lower() in ["true", "t", "1"]: return True
    elif v.lower() in ["false", "f", "0"]: return False
    else: raise argparse.ArgumentTypeError("Not Boolean value")

exitToken = 2
def ExitCounter():
    global exitToken
    exitToken -= 1
    if exitToken == 0:
        sys.exit()

def handler(signum, frame):
    # Log.Exit()
    print('Quit by user', signum)
    sys.exit()

class Stat():
    def __init__(self, args):
        self.loss = []
        self.accuracy = []
        self.running_loss = 0.0
        self.loss_count = 0
        self.loss_batches = args.stat_loss_batches

    def AddLoss(self, v):
        self.running_loss += v
        self.loss_count += 1
        if self.loss_count == self.loss_batches:
            self.loss.append(self.running_loss / self.loss_batches)
            self.loss_count = 0
            self.running_loss = 0.0
    
    def AddAccuracy(self, v):
        self.accuracy.append(v)

    def SaveToFile(self):
        f = open(args.log_file_location[:-4]+".stat", mode="w", newline='', encoding='utf-8')
        f.write(">Loss\n")
        for i in self.loss:
            f.write(str(i)+"\t")
        f.write("\n")
        f.write(">Accuracy\n")
        for i in self.accuracy:
            f.write(str(i)+"\t")
        f.write("\n")



def ArgumentParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--dataset", type=str, default = "CIFAR-10",
        help = "Dataset to use [CIFAR-10, CIFAR-100]")
    parser.add_argument("-m","--model", type=str, default = "Resnet18",
        help = "Model to use [SimpleNet, Resnet18]")
    parser.add_argument("-b","--block", type=str2bool, default=True,
        help = "Use Blocking [True False]")
    parser.add_argument("-c","--cuda", type=str2bool, default=True,
        help = "Using CUDA to compute on GPU [True False]")

    # Training setup
    parser.add_argument("-nw","--num-workers", type=int, default = 8,
        help = "Number of workers to load data")
    parser.add_argument("-gm","--group-mantissa", type=int, default = 8,
        help = "Group block's mantissa bit, default=8")
    parser.add_argument("-gs","--group-size", type=int, default = 36,
        help = "Group block's size, default=36")
    parser.add_argument("-gd","--group-direction", type=str, default = None,
        help = "Group block's grouping direction, Not implemented")

    # Printing / Logger / Stat
    parser.add_argument("-pti","--print-train-interval", type=int, default = 0,
        help = "Print interval, 0 to disable, both -pti and -ptc needed to be 0, -ptc is priortized") # 128 = 391
    parser.add_argument("-ptc","--print-train-count", type=int, default = 5,
        help = "How many print on each epoch, 0 to disable, both -pti and -ptc needed to be 0. -ptc is priortized") # 128 = 391
    parser.add_argument("-s","--stat", type=str2bool, default = True,
        help = "Record to stat object?")
    parser.add_argument("-slb","--stat-loss-batches", type=int, default = 100,
        help = "Average batches to calculate running loss on stat object")

    args = parser.parse_args()
    return args



def Evaluate(net, args, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if args.cuda:
                images = images.cuda() # Using GPU
                labels = labels.cuda() # Using GPU
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    Log.Print('Test Accuracy: %f' % (correct / total))
    if args.stat:
        args.stat_object.AddAccuracy(correct / total)

if __name__ == '__main__':
    # handle signal
    signal.signal(signal.SIGINT, handler)
    
    # Set logger preset
    Log.SetLogFile(True)
    # Parse Arguments
    args = ArgumentParse()
    # Save log file location
    args.log_file_location = Log.logFileLocation

    # Load dataset
    trainset, testset, classes = LoadDataset(args.dataset)

    criterion, optimizer, scheduler = None, None, None
    # Define the network and optimize almost everything
    if args.model == "SimpleNet":
        if args.block:
            net = BFSimpleNet(group_mantissa=args.group_mantissa, 
                group_size=args.group_size,
                group_direction=args.group_direction, 
                num_classes = len(classes))
        else:
            net = SimpleNet(num_classes = len(classes))
        if args.cuda:
            net.to('cuda')
            net = torch.nn.DataParallel(net) # Using GPU
        # Sometimes, there need to optimize optimizer and criterion
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=args.num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,shuffle=False, num_workers=args.num_workers)
        args.training_epochs = 5
    elif args.model == "Resnet18":
        # TODO : CIFAR-100 will not work
        if args.block:
            net = BFResNet18()
        else:
            net = ResNet18()
        # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
        if args.cuda:
            net.to('cuda')
            net = torch.nn.DataParallel(net) # Using GPU
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1,
                            momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer, milestones=[10, 20, 30, 40], gamma=0.1)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,shuffle=False, num_workers=args.num_workers)
        args.training_epochs = 200
    else:
        raise NotImplementedError("Model {} not Implemented".format(args.model))

    # Count of the mini-batches
    args.batch_count = len(trainloader)
    if args.print_train_count != 0:
        args.print_train_interval = math.ceil(args.batch_count / args.print_train_count)
    # Stat object
    if args.stat:
        stat = Stat(args)
        args.stat_object = stat


    # Print the model summary and arguments
    Log.Print("List of the training arguments\n" + str(args) + "\n", current=False, elapsed=False)
    Log.Print("Model Summary:%s"%net,current=False, elapsed=False)

    # Train the network
    for epoch_current in range(args.training_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            if args.cuda:
                inputs = inputs.cuda() # Using GPU
                labels = labels.cuda() # Using GPU
            
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if args.print_train_interval != 0 and ((i + 1) % args.print_train_interval == 0 or (i + 1) == len(trainloader)):    # print every 2000 mini-batches
                Log.Print('[%d/%d, %5d/%5d] loss: %.3f' %
                    (epoch_current + 1, args.training_epochs, i + 1, len(trainloader), running_loss / args.print_train_interval))
                running_loss = 0.0
            
            if args.stat:
                args.stat_object.AddLoss(loss.item())
        if scheduler != None:
            scheduler.step()
        Evaluate(net, args, testloader)
    Log.Print('Finished Training')

    if args.stat:
        args.stat_object.SaveToFile()


""" save model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
"""
