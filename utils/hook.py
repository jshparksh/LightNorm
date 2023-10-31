import torch.nn as nn
import torch.nn.functional as F
from bfp.module import BatchNorm2d_custom, RangeBN

class Hook():
    def __init__(self, module, forward=True):
        if forward:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()

def batchnorm_hook_result(model, input_data, input_label, optimizer):
    optimizer.zero_grad()
    model.train()
    forward_hook_list = []
    backward_hook_list = []
    
    for i, (name, module) in enumerate(model.named_modules()):
        if (isinstance(module, nn.BatchNorm2d)) and "shortcut" not in name: #BatchNorm2d_custom)) and "shortcut" not in name:
            temp_fwd_hook = Hook(module)
            temp_bwd_hook = Hook(module, forward=False)
            forward_hook_list.append(temp_fwd_hook)
            backward_hook_list.append(temp_bwd_hook)
    
    predict = model(input_data)
    loss = F.cross_entropy(predict, input_label)
    loss.backward()
    return forward_hook_list, backward_hook_list