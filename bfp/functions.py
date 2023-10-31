"""
    This code is part of the BFPSim (https://github.com/ids-Lab-DGIST/BFPSim)

    Seunghyun Lee (R3C0D3r) from IDSLab, DGIST
    coder@dgist.ac.kr

    License: CC BY 4.0
"""

import torch
import os
import json

from bfp.conf import BFPConf
from bfp.module import BFPLinear, BFPConv2d, BFPRangeBatchNorm2d, RangeBatchNorm2d_custom, RangeBatchNorm2d_custom_fwd, BFPRangeBatchNorm2d_custom_fwd, BatchNorm2d_custom, BFPBatchNorm2d_custom, RangeBN

# If you are migrating this code to separate library, plase remove line with "Log.Print" and next line
from utils.logger import Log

CONF_NET_PATH = "./conf_net/"

def LoadBFPDictFromFile(file):
    if file == "":
        Log.Print("bf layer config file not set, returning empty dict. This will generate unchanged model", current=False, elapsed=False)
        conf = dict()
    elif not os.path.exists(CONF_NET_PATH+file+".json"):
        raise FileNotFoundError("%s.json not exists on %s directory!"%(file, CONF_NET_PATH))
        # Log.Print(file + ".json not found, returning empty bf_conf_dict...", current=False, elapsed=False)
        # return dict()
    else:
        with open(CONF_NET_PATH+file+".json","r",encoding="utf-8") as f:
            conf = json.load(f)
    return conf

def GetValueFromDict(bfp_dict, attr_str):
    if attr_str in bfp_dict: # Layer configuration is found
        # If type is normal Conv2d
        if "type" in bfp_dict[attr_str] and bfp_dict[attr_str]["type"] in ["torch.nn.Conv2d", "torch.nn.Linear"]:
            return None  
        else:   # Found Config!
            return BFPConf(bfp_dict[attr_str])
    elif "default" in bfp_dict: # If default value is set, use the default value
        return BFPConf(bfp_dict["default"])
    else: # If no default value is set, don't replace
        return None

def ReturnBFPConv2d(ta, bfpc):
    if bfpc == None:
        return None
    bias = True if ta.bias != None else False
    new = BFPConv2d(in_channels=ta.in_channels, out_channels=ta.out_channels, kernel_size=ta.kernel_size, bfp_conf=bfpc, stride=ta.stride, padding=ta.padding, dilation=ta.dilation, groups=ta.groups, bias=bias, padding_mode=ta.padding_mode)
    return new

def ReturnBFPLinear(ta, bfpc):
    if bfpc == None:
        return None
    bias = True if ta.bias != None else False
    new = BFPLinear(in_features = ta.in_features, out_features=ta.out_features, bfp_conf=bfpc, bias=bias)
    return new

<<<<<<< HEAD
def ReturnBFPRangeBatchNorm2d(ta, bfpc):
    if bfpc == None:
        return None
    bias = True if ta.bias != None else False
    new = BFPRangeBatchNorm2d(ta.num_features, bfp_conf=bfpc, momentum=ta.momentum, num_chunks = 8, eps=ta.eps, beta=bias)
    return new

def ReturnRangeBatchNorm2d(ta, dtype):
    return RangeBN(ta.num_features, momentum=ta.momentum, eps=ta.eps)
    #return RangeBatchNorm2d_custom(ta.num_features, dtype, momentum=ta.momentum, eps=ta.eps)

def ReturnRangeBatchNorm2d_fwd(ta, dtype):
    #return RangeBN(ta.num_features, bfp_conf=bfpc, momentum=ta.momentum, eps=ta.eps)
    return RangeBatchNorm2d_custom_fwd(ta.num_features, dtype, momentum=ta.momentum, eps=ta.eps)

def ReturnBFPRangeBatchNorm2d_fwd(ta, dtype, bfpc):
    #return RangeBN(ta.num_features, bfp_conf=bfpc, momentum=ta.momentum, eps=ta.eps)
    return BFPRangeBatchNorm2d_custom_fwd(ta.num_features, dtype, bfpc, momentum=ta.momentum, eps=ta.eps)

def ReturnBFPBatchNorm2d(ta, bfpc):
    return BFPBatchNorm2d_custom(ta.num_features, bfp_conf=bfpc, momentum=ta.momentum, eps=ta.eps)
 
def ReturnBatchNorm2d(ta, dtype):
    return BatchNorm2d_custom(ta.num_features, dtype, momentum=ta.momentum, eps=ta.eps) #BatchNorm(ta.num_features, dtype, momentum=ta.momentum, eps=ta.eps)

    
def _ReplaceInternal(net, name, attr_str, attr_value, bfpc, dtype, mode):
    """if type(attr_value) == torch.nn.Conv2d: # Conv2d is replaced
        Log.Print("Detected %s : %s"%(name+"."+attr_str, attr_value), current=False, elapsed=False)
        if bfpc == None:
            Log.Print("  == Didn't replaced", current=False, elapsed=False)
        else:
            if mode == "C":
                setattr(net, attr_str, ReturnBFPConv2d(attr_value, bfpc))
            elif mode == "S":
                net[int(attr_str)] = ReturnBFPConv2d(net[int(attr_str)], bfpc)
            else:
                raise ValueError("Replace Method is not supported.")
            Log.Print("  => Replaced to BFPConv2d:%s"%(str(bfpc)), current=False, elapsed=False)"""
    
    if type(attr_value) == torch.nn.BatchNorm2d:
        Log.Print("Detected %s : %s"%(name+"."+attr_str, attr_value), current=False, elapsed=False)
        if bfpc == None:
            Log.Print("  == Didn't replaced", current=False, elapsed=False)
        else:
            if mode == "C":
                #setattr(net, attr_str, ReturnBFPRangeBatchNorm2d(attr_value, bfpc))
                #setattr(net, attr_str, ReturnRangeBatchNorm2d(attr_value, bfpc))
                #setattr(net, attr_str, ReturnBatchNorm2d(attr_value, dtype))
                #setattr(net, attr_str, ReturnRangeBatchNorm2d_fwd(attr_value, dtype))
                setattr(net, attr_str, ReturnBFPRangeBatchNorm2d_fwd(attr_value, dtype, bfpc))
            elif mode == "S":
                #net[int(attr_str)] = ReturnBFPRangeBatchNorm2d(net[int(attr_str)], bfpc)
                #net[int(attr_str)] = ReturnBatchNorm2d(net[int(attr_str)], bfpc)
                #net[int(attr_str)] = ReturnBatchNorm2d(net[int(attr_str)], dtype)
                #net[int(attr_str)] = ReturnRangeBatchNorm2d_fwd(net[int(attr_str)], dtype)
                net[int(attr_str)] = ReturnBFPRangeBatchNorm2d_fwd(net[int(attr_str)], dtype, bfpc)
            else:
                raise ValueError("Replace Method is not supported.")
            #Log.Print("  => Replaced to BFPRangeBatchNorm2d:%s"%(str(bfpc)), current=False, elapsed=False)
            #Log.Print("  => Replaced to BatchNorm2d:%s"%(dtype), current=False, elapsed=False)
            #Log.Print("  => Replaced to RangeBatchNorm2d_fwd:%s"%(dtype), current=False, elapsed=False)
            Log.Print("  => Replaced to BFPRangeBatchNorm2d:%s"%(str(bfpc)), current=False, elapsed=False) 

    if type(attr_value) == torch.nn.Linear: # Linear is replaced
        return
        # TODO : Fix here 
        Log.Print("Detected %s : %s"%(name+"."+attr_str, attr_value), current=False, elapsed=False)
        if bfpc == None:
            Log.Print("  == Didn't replaced", current=False, elapsed=False)
        else:
            if mode == "C":
                setattr(net, attr_str, ReturnBFPLinear(attr_value, bfpc))
            elif mode == "S":
                net[i] = ReturnBFPLinear(net[i], bfpc)
            else:
                raise ValueError("Replace Method is not supported.")
            Log.Print("  => Replaced to BFPLinear:%s"%(str(bfpc)), current=False, elapsed=False)
    

def ReplaceLayers(net, bfp_dict, dtype='fp32', name="net"):
    # Replace child objects
=======
def GetValueFromDict(bfp_dict, attr_str):
    if attr_str in bfp_dict: # Layer configuration is found
        if "type" in bfp_dict[attr_str] and bfp_dict[attr_str]["type"] in ["torch.nn.Conv2d", "torch.nn.Linear", "default"]:
            return None
        else:   # Found Config!
            return BFPConf(bfp_dict[attr_str])
    elif "default" in bfp_dict: # If default value is set, use the default value
        return BFPConf(bfp_dict["default"])
    else: # If no default value is set, don't replace
        return None

def ReplaceLayers(net, bfp_dict, name="net", silence = False):
    # Log.Print("%s / %s"%(type(net),name), current=False, elapsed=False)
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
    for attr_str in dir(net):
        # Get the Attributes
        attr_value = getattr(net, attr_str)
<<<<<<< HEAD
        bfpc = GetValueFromDict(bfp_dict, name+"."+attr_str)
        _ReplaceInternal(net, name, attr_str, attr_value, bfpc, dtype, mode="C")

    # Replacing List, sequential, etc
    for attr_idx, attr_value in enumerate(net.children()):
        bfpc = GetValueFromDict(bfp_dict, name+"."+attr_str)
        _ReplaceInternal(net, name, str(attr_idx), attr_value, bfpc, dtype, mode="S")
            

    # Recursive call to replace other layers
    for n, ch in net.named_children():
        ReplaceLayers(ch, bfp_dict, dtype, name+"."+n)
    if type(net) in [list, tuple, torch.nn.Sequential]:
        for i, n in enumerate(net.children()):
            ReplaceLayers(net[i], bfp_dict, dtype, name+"."+str(i))
=======
        if attr_str == "zero_grad":
            continue
        if type(attr_value) in [torch.nn.Conv2d, torch.nn.Linear]:
            if not silence:
                Log.Print("Detected(N) %s : %s"%(name+"."+attr_str, attr_value), current=False, elapsed=False)
            bfpc = GetValueFromDict(bfp_dict, name+"."+attr_str)
            if bfpc != None:
                # Replace Actual
                if type(attr_value) == torch.nn.Conv2d:
                    setattr(net, attr_str, ReturnBFPConv2d(attr_value, bfpc))
                elif type(attr_value) == torch.nn.Linear:
                    setattr(net, attr_str, ReturnBFPLinear(attr_value, bfpc))
                if not silence:
                    Log.Print("  => Replaced : %s"%(str(bfpc)), current=False, elapsed=False)
            else:
                if not silence:
                    Log.Print("  == Didn't replaced", current=False, elapsed=False)
    
    # Log.Print("Child @ %s"%name, current=False, elapsed=False)
    for n, ch in net.named_children():
        ReplaceLayers(ch, bfp_dict, name+"."+n, silence = silence)
    # Log.Print("Iter @ %s"%name, current=False, elapsed=False)
    if type(net) in [list, tuple, torch.nn.Sequential]:
        for i, n in enumerate(net.children()):
            if type(net[i]) in [torch.nn.Conv2d, torch.nn.Linear]:
                if not silence:
                    Log.Print("Detected(I) %s : %s"%(name+"."+str(i), n), current=False, elapsed=False)
                bfpc = GetValueFromDict(bfp_dict, name+"."+str(i))
                if bfpc != None:
                    # Replace Actual
                    # TODO : Check if this is works or not
                    if type(n) == torch.nn.Conv2d:
                        net[i] = ReturnBFPConv2d(n, bfpc)
                    elif type(n) == torch.nn.Linear:
                        net[i] = ReturnBFPLinear(n, bfpc)
                    if not silence:
                        Log.Print("  => Replaced : %s"%(str(bfpc)), current=False, elapsed=False)
                else:
                    if not silence:
                        Log.Print("  == Didn't replaced", current=False, elapsed=False)

            ReplaceLayers(net[i], bfp_dict, name+"."+str(i), silence = silence)

    # Log.Print("End @ %s"%name, current=False, elapsed=False)

def GetBFLayerNames(net, name="net"):
    # print(type(net), name)
    res = []
    for n, ch in net.named_children():
        if type(ch) in [BFPConv2d, BFPLinear]:
            # Log.Print("Detected(N) %s : %s"%(name+"."+n, ch), current=False, elapsed=False)
            res.append(name+"."+n)
        l = GetBFLayerNames(ch, name + "." + n)
        for i in l:
            res.append(i)
    return res
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b


"""
Example of Using ReplaceLayers()

# It is possible to make bpf_dict by own
bfp_dict = dict()
# Define default value
bfp_dict["default"] = BFPConf()
bfp_dict["net.conv1"] = BFPConf()
ReplaceLayers(net, bfp_dict)

# Or, you can provide the file's path
path = "default_FB12_WG24"
bfp_dict = LoadBFPDictFromFile(path)
ReplaceLayers(net, bfp_dict)

"""
