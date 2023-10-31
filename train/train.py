from matplotlib.pyplot import flag
import torch
import torch.optim as optim
import os
from utils.logger import Log
from utils.slackBot import slackBot
from utils.statManager import statManager
from utils.save import SaveModel
from utils.hook import batchnorm_hook_result
from train.network import GetNetwork, GetOptimizer, GetScheduler
from bfp.functions import LoadBFPDictFromFile
import wandb

def TrainMixed(args, epoch_current):
    running_loss = 0.0
    batch_count = 0
    ptc_count = 1
    ptc_target = ptc_count / args.print_train_count


    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        for i, data in enumerate(args.trainloader, 0):
            inputs, labels = data
            
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            args.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                outputs = args.net(inputs)
                assert outputs.dtype is torch.float16

                loss = args.criterion(outputs, labels)
                assert loss.dtype is torch.float32

            # loss.backward()
            args.scaler.scale(loss).backward()
            args.scaler.step(args.optimizer)
            args.scaler.update()

            running_loss += loss.item()

            # Print the running loss
            pF = False
            batch_count += 1
            if args.print_train_batch != 0:
                if (i + 1) % args.print_train_batch == 0 or (i + 1) == len(args.trainloader):
                    pF = True
            elif args.print_train_count != 0:
                if (i + 1) / len(args.trainloader) >= ptc_target:
                    pF = True
                    ptc_count += 1
                    ptc_target = ptc_count/args.print_train_count
            if pF:
                Log.Print('[%d/%d, %5d/%5d] loss: %.3f' %
                    (epoch_current + 1, args.training_epochs,
                    i + 1, len(args.trainloader),
                    running_loss / batch_count))
                running_loss = 0.0
                batch_count = 0

import pickle 

def Train(args, epoch_current):
    running_loss = 0.0
    batch_count = 0
    ptc_count = 1
    ptc_target = ptc_count / args.print_train_count
    
    top1, top5, total = 0, 0, 0
    
    random_sampler= torch.utils.data.RandomSampler(args.testloader.dataset)
    sample_loader = torch.utils.data.DataLoader(args.testloader.dataset, batch_size=128, sampler=random_sampler)
    check_data, check_label = next(iter(sample_loader))
    check_data = check_data.to('cuda')
    check_label = check_label.to('cuda')
    result_fwd_hook_list, result_bwd_hook_list = batchnorm_hook_result(args.net, check_data, check_label, args.optimizer)
    
    range = {'input':[0,0],'output':[0,0],'grad_in':[0,0],'grad_out':[0,0]}
    for hook in result_fwd_hook_list:
        batch_input = hook.input[0].cpu().detach()
        batch_output = hook.output.cpu().detach()
        range['input'][0] = torch.min(batch_input)
        range['input'][1] = torch.max(batch_input)
        range['output'][0] = torch.min(batch_output)
        range['output'][1] = torch.max(batch_output)
        hook.close()
    for hook in result_bwd_hook_list:
        batch_input = hook.input[0].cpu().detach()
        batch_output = hook.output[0].cpu().detach()
        range['grad_in'][0] = torch.min(batch_input)
        range['grad_in'][1] = torch.max(batch_input)
        range['grad_out'][0] = torch.min(batch_output)
        range['grad_out'][1] = torch.max(batch_output)
        hook.close()
    save_range_path = os.path.join("./_ranges", f"{args.model}_ranges.txt")
    if not os.path.exists(os.path.join("./_ranges")):
        os.makedirs(os.path.join("./_ranges"))
    with open(save_range_path, "a") as fw:
        fw.write(str(epoch_current)+': '+str(range))
        
    # To save forward and backward data
    """
    #activation_step = [0, 40, 80, 120, 160, 200]
    #if epoch_current in activation_step:
    result_fwd_hook_list, result_bwd_hook_list = batchnorm_hook_result(args.net, check_data, check_label, args.optimizer)
    for hook in result_fwd_hook_list:
        batch_input = hook.input[0].cpu().detach()
        batch_output = hook.output.cpu().detach()
        #print(f"batch_input shape : {batch_input.shape}")
        #print(f"batch_output shape : {batch_output.shape}")
        save_input_pkl_path = os.path.join(args.pickle_path, f"{args.model}_{epoch_current}_new_fwd_input.pkl")
        if not os.path.exists(os.path.join(args.pickle_path)):
            os.makedirs(os.path.join(args.pickle_path))
        save_output_pkl_path = os.path.join(args.pickle_path, f"{args.model}_{epoch_current}_new_fwd_act.pkl")                
        
        with open(save_input_pkl_path, "wb") as fw:
            pickle.dump(batch_input, fw)
        with open(save_output_pkl_path, "wb") as bw:
            pickle.dump(batch_output, bw)
        hook.close()
    
    for hook in result_bwd_hook_list:
        batch_input = hook.input[0].cpu().detach()
        batch_output = hook.output[0].cpu().detach()
        #print(f"batch_grad_input shape : {batch_input.shape}")
        #print(f"batch_grad_input tuples shape : {[len(f) for f in hook[1].input]}")
        #print(f"batch_grad_output shape : {batch_output.shape}")
        #print(f"batch_grad_output tuples shape : {[len(f) for f in hook[1].output]}")
        
        save_input_pkl_path = os.path.join(args.pickle_path, f"{args.model}_{epoch_current}_new_act_grad_in.pkl")   
        save_output_pkl_path = os.path.join(args.pickle_path, f"{args.model}_{epoch_current}_new_act_grad_out.pkl")                
        with open(save_input_pkl_path, "wb") as fw:
            pickle.dump(batch_input, fw)
        with open(save_output_pkl_path, "wb") as bw:
            pickle.dump(batch_output, bw)
        hook.close()
    """
    for i, data in enumerate(args.trainloader, 0):
        inputs, labels = data
        
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        args.optimizer.zero_grad()
        outputs = args.net(inputs)
        loss = args.criterion(outputs, labels)

        args.scaler.scale(loss).backward()
        args.scaler.step(args.optimizer)
        args.scaler.update()
        torch.nn.utils.clip_grad_norm_(args.net.parameters(), 5)

        running_loss += loss.item()
        
        # Print the running loss
        pF = False
        batch_count += 1
        if args.print_train_batch != 0:
            if (i + 1) % args.print_train_batch == 0 or (i + 1) == len(args.trainloader):
                pF = True
        elif args.print_train_count != 0:
            if (i + 1) / len(args.trainloader) >= ptc_target:
                pF = True
                ptc_count += 1
                ptc_target = ptc_count/args.print_train_count
        if pF:
            Log.Print('[%d/%d, %5d/%5d] loss: %.3f' %
                (epoch_current + 1, args.training_epochs,
                i + 1, len(args.trainloader),
                running_loss / batch_count))
            acc1, acc5 = Accuracy(outputs, labels, topk=(1, 5))
            top1 += acc1[0] * inputs.size(0)
            top5 += acc5[0] * inputs.size(0)
            total += inputs.size(0)
            Log.Print('[%d/%d, %5d/%5d] Train_step_acc(t1):%7.3f' %
                (epoch_current+1, args.training_epochs, 
                i + 1, len(args.trainloader),
                (top1/total).cpu().item()))
            running_loss = 0.0
            batch_count = 0
    
    Log.Print('[%d/%d], TrainAcc(t1):%7.3f' % (epoch_current+1, args.training_epochs, (top1/total).cpu().item()))    
    
        

"""
Accuracy Code from pytorch example
https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""
def Accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def Evaluate(args, mode = "test"):
    if mode == "test":
        loader = args.testloader
    elif mode == "train":
        loader = args.trainloader
    else:
        raise ValueError("Mode not supported")
    top1, top3, top5, total = 0, 0, 0, 0
    args.net.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = args.net(images)
            loss = args.criterion(output, target)
            acc1, acc3, acc5 = Accuracy(output, target, topk=(1, 3, 5))
            top1 += acc1[0] * images.size(0)
            top3 += acc3[0] * images.size(0)
            top5 += acc5[0] * images.size(0)
            total += images.size(0)
    return (top1/total).cpu().item(), (top3/total).cpu().item(), (top5/total).cpu().item()

# Train the network and evaluate
def TrainNetwork(args):
    Log.Print("========== Starting Training ==========")
    args.scaler = torch.cuda.amp.GradScaler() # FP16 Mixed Precision
    for epoch_current in range(args.start_epoch, args.training_epochs):
        # Change and transfer model
        """if epoch_current != args.start_epoch and str(epoch_current) in args.bfp_layer_conf_dict:
            Log.Print("Changing Model bfp config to: %s"%args.bfp_layer_conf_dict[str(epoch_current)], elapsed=False, current=False)
            net_ = args.net
            args.net = GetNetwork(args.dataset, args.model, args.num_classes, LoadBFPDictFromFile(args.bfp_layer_conf_dict[str(args.start_epoch)]))
            args.net.load_state_dict(net_.state_dict())
            args.net.eval()

            # Load Optimizer, Scheduler, stuffs
            args.optimizer = GetOptimizer(args, str(epoch_current))
            args.scheduler = GetScheduler(args, str(epoch_current))
            if args.cuda:
                args.net.to('cuda')
        """
        # Train the net
        #wandb.watch(args.net, log="all", log_freq=10)
        Train(args, epoch_current)
        # Evaluate the net
        t1, t3, t5 = Evaluate(args)
        
        statManager.AddData("top1test", t1)
        statManager.AddData("top3test", t3)
        statManager.AddData("top5test", t5)
        Log.Print('[%d/%d], TestAcc(t1):%7.3f, lr:%f' % (epoch_current+1, args.training_epochs, t1, args.optimizer.param_groups[0]['lr']))

        if args.scheduler != None:
            args.scheduler.step()

        # Save the model
        if args.save:
            if args.save_interval != 0 and (epoch_current+1)%args.save_interval == 0:
                SaveModel(args, "%03d"%(epoch_current+1))

    Log.Print("========== Finished Training ==========")
        
    if args.stat:
        Log.Print("Saving stat object file...")
        statManager.SaveToFile(args.stat_location)

    if args.save:
        SaveModel(args, "finish")
