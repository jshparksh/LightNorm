import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from typing import TypeVar, Union, Tuple, Optional

T = TypeVar('T')
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_scalar_or_tuple_3_t = Union[T, Tuple[T, T, T]]

_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]
_size_3_t = _scalar_or_tuple_3_t[int]

from bfp.internal import make_groups_tensor, set_precision
from bfp.conf import BFPConf

# BlockFloat Linear Function
class BFPLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, bfp_conf):    
        # Grouping input and weight
        if bfp_conf.fi:
            input = make_groups_tensor(input, bfp_conf.fi_bit, bfp_conf.fi_dim)
        if bfp_conf.fw:
            weight = make_groups_tensor(weight, bfp_conf.fw_bit, bfp_conf.fw_dim)
        # Save context to use on backward
        ctx.bfp_conf = bfp_conf
        ctx.save_for_backward(input, weight, bias)
        
        # Compute FC and return
        output = F.linear(input, weight, bias)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        # Grouping Output
        if bfp_conf.fo:
            output = make_groups_tensor(output, bfp_conf.fo_bit, bfp_conf.fo_dim)
        print("F: %s %s %s"%(input.shape, weight.shape, output.shape))
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        # input, weight, bias, confs = ctx.saved_tensors
        input, weight, bias = ctx.saved_tensors
        bfp_conf = ctx.bfp_conf
        print("output gradient")

        # Calculate gradients
        grad_input = grad_weight = grad_bias = None

        # Calculate Input Gradient
        ## Grouping grad_output
        if bfp_conf.bio:
            grad_output_ = make_groups_tensor(grad_output, bfp_conf.bio_bit, bfp_conf.bio_dim)
        else: # Apply original gradient if grad_output is not grouped
            grad_output_ = grad_output
        ## Grouping weight
        if bfp_conf.biw:
            weight = make_groups_tensor(weight, bfp_conf.biw_bit, bfp_conf.biw_dim)        

        print(grad_output_.shape)
        print(weight.shape)
        grad_input_ = F.linear(grad_output_, weight.t(), bias).t()
        print(grad_input.shape)
        # grad_input_ = grad_output_.mm(weight)

        if bfp_conf.big:
            grad_input_ = make_groups_tensor(grad_input, bfp_conf.big_bit,bfp_conf.big_dim)
        else: # If not grouping, use original type
            grad_input_ = grad_input

        if bfp_conf.bwo:
            # Regroup if bwo / bio grouping configuration is different!
            if (bfp_conf.bwo_bit != bfp_conf.bio_bit or bfp_conf.bwo_dim != bfp_conf.bio_dim):
                grad_output_ = make_groups_tensor(grad_output, bfp_conf.bwo_bit, bfp_conf.bwo_dim)
        else: # If not grouping, use original type
            grad_output_ = grad_output
        ## Grouping input - it's not grad_input, right?
        if bfp_conf.bwi:
            # Regroup if bwi / fi grouping configuration is different!
            if (bfp_conf.bwi_bit != bfp_conf.fi_bit or bfp_conf.bwi_dim != bfp_conf.fi_dim):
                input = make_groups_tensor_fc(input, bfp_conf.bwi_bit, bfp_conf.bwi_dim)
        grad_weight = F.linear(grad_output_.t(), input.t(), bias).t()
        # grad_weight = grad_output_.t().mm(input)
        # print(grad_weight.shape)
        # Group the gradient of weight
        if bfp_conf.bwg:
            grad_weight = make_groups_tensor_fc(grad_weight, bfp_conf.bwg_bit, bfp_conf.bwg_dim)

        if bfp_conf.bwg_boost != 1.0:
            grad_weight /= bfp_conf.bwg_boost

        if bias is not None:
            grad_bias = grad_output.sum(0)

        return grad_input_, grad_weight, grad_bias, None

# Blockfloat Linear
class BFPLinear(torch.nn.Module):
    def __init__(self,
                input_features: int,
                output_features: int,
                bfp_conf: BFPConf,
                bias=True):
        super(BFPLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.bfp_conf = bfp_conf

        # Weight parameters, should be grouped with few numbers
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_paramter('bias', None)
        
        # Initialize weights manually
        self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
    
    def forward(self, input):
        return BFPLinearFunction.apply(input, self.weight, self.bias, self.bfp_conf)
    
    def extra_repr(self):
        s = ('{input_features}, {output_features}')
        s += ', bfp_conf=({bfp_conf})'
        if self.bias is None:
            s += ', bias=False'
        else:
            s += ', bias=True'
        return s.format(**self.__dict__)


# Blockfloat Convolution Function
# TODO : Implement Conv2d Operation
# https://discuss.pytorch.org/t/implementing-a-custom-convolution-using-conv2d-input-and-conv2d-weight/18556/7
class BFPConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, bfp_conf=None, stride=1, padding=0, dilation=1, groups=1):
        # print("= Forward:",input.shape, weight.shape, stride, padding, dilation, groups)
        # Grouping input and weight
        if bfp_conf.fi:
            input_ = make_groups_tensor(input.clone().detach(), bfp_conf.fi_bit, bfp_conf.fi_dim, 0)
        else:
            input_ = input
        if bfp_conf.fw:
            weight_ = make_groups_tensor(weight.clone().detach(), bfp_conf.fw_bit, bfp_conf.fw_dim, 1)
        else:
            weight_ = weight

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.bfp_conf = bfp_conf

        ctx.save_for_backward(input_, weight_, bias)

        # Compute Convolution
        output = F.conv2d(input_, weight_, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        # Grouping Output
        if bfp_conf.fo:
            output = make_groups_tensor(output.clone().detach(), bfp_conf.fo_bit, bfp_conf.fo_dim, 2)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors and configs
        input, weight, bias = ctx.saved_variables
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        bfp_conf = ctx.bfp_conf

        # print("= Backward:",grad_output.shape, stride, padding, dilation, groups)
        grad_input = grad_weight = grad_bias = None

        # Calculate Input Gradient
        ## Grouping grad_output
        if bfp_conf.bio:
            grad_output_ = make_groups_tensor(grad_output.clone().detach(), bfp_conf.bio_bit, bfp_conf.bio_dim, 10)
        else: # Apply original gradient if grad_output is not grouped
            grad_output_ = grad_output
        ## Grouping weight
        if bfp_conf.biw:
            if (bfp_conf.biw_bit != bfp_conf.fw_bit or bfp_conf.biw_dim != bfp_conf.fw_dim):
                weight_ = make_groups_tensor(weight.clone().detach(), bfp_conf.biw_bit, bfp_conf.biw_dim, 11)
            else:
                weight_ = weight
        else:
            weight_ = weight
        ## Do the convolution
        if ctx.needs_input_grad[0]: # First Layer's grad_input will be None
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight_, grad_output_, stride, padding, dilation, groups)
        ## Grouping output grad_input
        if bfp_conf.big and grad_input != None:
            grad_input_ = make_groups_tensor(grad_input.clone().detach(), bfp_conf.big_bit,bfp_conf.big_dim, 12)
        else: # If not grouping, use original type
            grad_input_ = grad_input

        # Calculate Weight Gradient (2D Convolution, Depthwise Convolution)
        ## Grouping grad_output
        
        if bfp_conf.bwo:
            # Regroup if bwo / bio grouping configuration is different!
            if (bfp_conf.bwo_bit != bfp_conf.bio_bit or bfp_conf.bwo_dim != bfp_conf.bio_dim):
                grad_output_ = make_groups_tensor(grad_output.clone().detach(), bfp_conf.bwo_bit, bfp_conf.bwo_dim, 20)
        else: # If not grouping, use original type
            grad_output_ = grad_output
        ## Grouping input - it's not grad_input, right?
        if bfp_conf.bwi:
            # Regroup if bwi / fi grouping configuration is different!
            if (bfp_conf.bwi_bit != bfp_conf.fi_bit or bfp_conf.bwi_dim != bfp_conf.fi_dim):
                input = make_groups_tensor(input.clone().detach(), bfp_conf.bwi_bit, bfp_conf.bwi_dim, 21)
        ## Do the convolution
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output_, stride, padding, dilation, groups)
        # Group the gradient of weight
        if bfp_conf.bwg and grad_weight != None:
            grad_weight = make_groups_tensor(grad_weight.clone().detach(), bfp_conf.bwg_bit, bfp_conf.bwg_dim, 22)

        # Apply weaken gradient if weight gradient boost is applied
        if bfp_conf.bwg_boost != 1.0:
            grad_weight /= bfp_conf.bwg_boost

        # TODO : Add Bias Grouping / or is it needed?
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_.sum(dim=(0,2,3)).squeeze(0)
        
        return grad_input_, grad_weight, grad_bias, None, None, None, None, None

# Blockfloat Convolution
class BFPConv2d(torch.nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: _size_2_t,
                bfp_conf: BFPConf, 
                stride: _size_2_t = 1,
                padding: _size_2_t = 0,
                dilation: _size_2_t = 1,
                groups: int = 1,
                bias: bool = True,
                padding_mode: str = 'zeros'):
        super(BFPConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        # TODO : Edit this area
        if type(kernel_size) == int:
            self.kernel_size = kernel_size
        else:
            self.kernel_size = kernel_size[0]

        self.bfp_conf = bfp_conf
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, self.kernel_size, self.kernel_size))
        # self.bias = nn.Parameter(torch.Tensor(out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return BFPConv2dFunction.apply(input, self.weight, self.bias, self.bfp_conf, self.stride, self.padding, self.dilation, self.groups)
    
    def extra_repr(self):
        # From /torch/nn/modules/conv.py
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        s += ', bfp_conf=({bfp_conf})'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

def unsqueeze_all(t):
    # Helper function to unsqueeze all the dimensions that we reduce over
    return t[None, :, None, None]

class RangeBN(nn.Module):
    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=8, eps=1e-5):
        super(RangeBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        input_ = x
        gamma_ = self.weight
        #if self.training:
        B, C, H, W = input_.shape
        y = input_.transpose(0, 1).contiguous()  # C x B x H x W
        y = y.view(C, self.num_chunks, B * H * W // self.num_chunks)
        mean_max = y.max(-1)[0].mean(-1)  # C
        mean_min = y.min(-1)[0].mean(-1)  # C
        mean = y.view(C, -1).mean(-1)  # C
        #scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
        #                            0.5) / ((2 * math.log(y.size(-1))) ** 0.5)
        scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
        scale = 1 / ((mean_max - mean_min) * scale_fix + self.eps)
        #print('scale', scale)
        self.running_mean.detach().mul_(self.momentum).add_(
            mean * (1 - self.momentum))

        self.running_var.detach().mul_(self.momentum).add_(
            scale * (1 - self.momentum))
        """else:
            mean = self.running_mean
            scale = self.running_var"""
            
        out = (x - mean.view(1, mean.size(0), 1, 1)) * \
            scale.view(1, scale.size(0), 1, 1)
        out = out * gamma_.view(1, gamma_.size(0), 1, 1) + self.bias.view(1, self.bias.size(0), 1, 1)

        return out

class RangeBN_bfp(nn.Module):
    def __init__(self, num_features, bfp_conf, dim=1, momentum=0.1, affine=True, num_chunks=8, eps=1e-5):
        super(RangeBN_bfp, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.bfp_conf = bfp_conf
        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        if self.bfp_conf.fi:
            #input_ = make_groups_tensor(x.clone().detach(), self.bfp_conf.fi_bit, self.bfp_conf.fi_dim, 0)
            input_ = x
        else:
            input_ = x
        if self.bfp_conf.fw:
            gamma_ = self.weight
            #gamma_ = make_groups_tensor(self.weight.clone().detach(), self.bfp_conf.fw_bit, self.bfp_conf.fw_dim, 1)
        else:
            gamma_ = self.weight
        #input_ = set_precision(x.clone().detach(), "bfloat16")
        #if self.training:
        B, C, H, W = input_.shape
        y = input_.transpose(0, 1).contiguous()  # C x B x H x W
        y = y.view(C, self.num_chunks, B * H * W // self.num_chunks)
        mean_max = y.max(-1)[0].mean(-1)  # C
        mean_min = y.min(-1)[0].mean(-1)  # C
        mean = y.view(C, -1).mean(-1)  # C
        #scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
        #                            0.5) / ((2 * math.log(y.size(-1))) ** 0.5)
        scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
        scale = 1 / ((mean_max - mean_min) * scale_fix + self.eps)
        #print('scale', scale)
        self.running_mean.detach().mul_(self.momentum).add_(
            mean * (1 - self.momentum))

        self.running_var.detach().mul_(self.momentum).add_(
            scale * (1 - self.momentum))
        """else:
            mean = self.running_mean
            scale = self.running_var"""
            
        out = (x - mean.view(1, mean.size(0), 1, 1)) * \
            scale.view(1, scale.size(0), 1, 1)
        out = out * gamma_.view(1, gamma_.size(0), 1, 1) + self.bias.view(1, self.bias.size(0), 1, 1)

        return out

class BFPRangeBatchNorm2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, bfp_conf, num_chunks=8, eps=1e-5):    #input, weight, bias=None, bf_conf=None, stride=1, padding=0, dilation=1, groups=1):
        # print("= Forward:",input.shape, weight.shape, stride, padding, dilation, groups)
        # Grouping input and weight
        if bfp_conf.fi:
            input_ = input
            #input_ = make_groups_tensor(input.clone().detach(), bfp_conf.fi_bit, bfp_conf.fi_dim, 0)
        else:
            input_ = input
        if bfp_conf.fw:
            gamma_ = gamma
            #gamma_ = make_groups_tensor(gamma.clone().detach(), bfp_conf.fw_bit, bfp_conf.fw_dim, 1)
        else:
            gamma_ = gamma

        B, C, H, W = input_.shape
        y = input_.transpose(0, 1).contiguous()  # C x B x H x W
        y = y.view(C, num_chunks, B * H * W // num_chunks)
        avg_max = y.max(-1)[0].mean(-1)  # C
        avg_min = y.min(-1)[0].mean(-1)  # C
        avg = y.view(C, -1).mean(-1)  # C
        #scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
        #                            0.5) / ((2 * math.log(y.size(-1))) ** 0.5)
        scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
        scale = 1 / ((avg_max - avg_min) * scale_fix + eps)

        #running_mean.detach().mul_(momentum).add_(
        #    avg * (1 - momentum))

        #running_var.detach().mul_(momentum).add_(
        #    scale * (1 - momentum))
        # else:
        #     avg = running_mean
        #     scale = running_var

        ctx.avg = avg
        ctx.eps = eps
        ctx.scale_fix = scale_fix
        ctx.scale = scale
        ctx.num_chunks = num_chunks
        ctx.bfp_conf = bfp_conf

        avg = avg.view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1)
        output = (input_ - avg) * scale
        ctx.save_for_backward(input_, gamma_, beta, scale, output)
        output = output * gamma_ + beta

        #if bfp_conf.fo:
        #    output = make_groups_tensor(output.clone().detach(), bfp_conf.fo_bit, bfp_conf.fo_dim, 2)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors and configs
        input, gamma, beta, scale, output = ctx.saved_variables
        B, C, H, W = input.shape
        avg = ctx.avg
        scale_fix = ctx.scale_fix
        scale = ctx.scale
        num_chunks = ctx.num_chunks
        eps = ctx.eps
        bfp_conf = ctx.bfp_conf
        avg = unsqueeze_all(avg)
        scale = unsqueeze_all(scale)

        if bfp_conf.bio:
            grad_output_ = grad_output
            #grad_output_ = make_groups_tensor(grad_output.clone().detach(), bfp_conf.bio_bit, bfp_conf.bio_dim, 10)
        else: # Apply original gradient if grad_output is not grouped
            grad_output_ = grad_output
        ## Grouping gamma
        if bfp_conf.biw:
            if (bfp_conf.biw_bit != bfp_conf.fw_bit or bfp_conf.biw_dim != bfp_conf.fw_dim):
                gamma_ = gamma
                #gamma_ = make_groups_tensor(gamma.clone().detach(), bfp_conf.biw_bit, bfp_conf.biw_dim, 11)
            else:
                gamma_ = gamma
        else:
            gamma_ = gamma

        grad_input = gamma_ * (scale ** 2) * (scale_fix - 1/(B * num_chunks)) * grad_output_ 
        grad_gamma = (grad_output_ * output).sum(dim=(0, 2, 3), keepdim=True)
        grad_beta = (grad_output_).sum(dim=(0, 2, 3), keepdim=True)
        if bfp_conf.big and grad_input != None:
            grad_input_ = grad_input
            #grad_input_ = make_groups_tensor(grad_input.clone().detach(), bfp_conf.big_bit,bfp_conf.big_dim, 12)
        else: # If not grouping, use original type
            grad_input_ = grad_input

        #if bfp_conf.bwg and grad_weight != None:
        #    grad_gamma = make_groups_tensor(grad_gamma.clone().detach(), bfp_conf.bwg_bit, bfp_conf.bwg_dim, 22)

        return grad_input_, grad_gamma, grad_beta, None, None, None
        
        
# Blockfloat Batchnorm
class BFPRangeBatchNorm2d(torch.nn.Module):
    def __init__(self,
                num_features: int,
                bfp_conf: BFPConf,
                momentum = 0.1,
                num_chunks = 8,
                eps = 1e-5,
                beta: bool = True):
        super(BFPRangeBatchNorm2d, self).__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.num_chunks = num_chunks
        self.eps = eps
        self.bfp_conf = bfp_conf
        self.gamma = nn.Parameter(torch.Tensor(num_features))
        if beta:
            self.beta = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self): # -> None:
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.gamma.data.uniform_()
        self.beta.data.zero_()
        '''
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        '''
    def forward(self, input):
        gamma = self.gamma.view(1, self.num_features, 1, 1)
        beta = self.beta.view(1, self.num_features, 1, 1)
        return BFPRangeBatchNorm2dFunction.apply(input, gamma, beta, self.bfp_conf, self.num_chunks, self.eps)
        """ctx, input, gamma, beta, bfp_conf, running_mean, running_var, momentum, num_chunks=64, eps=1e-5, training=True
        gamma = self.weight.view(1, self.num_features, 1, 1)
        beta = self.beta.view(1, self.num_features, 1, 1)

        if self.bfp_conf.fi:
            input_ = make_groups_tensor(input.clone().detach(), self.bfp_conf.fi_bit, self.bfp_conf.fi_dim, 0)
        else:
            input_ = input
        if self.bfp_conf.fw:
            gamma_ = make_groups_tensor(gamma.clone().detach(), self.bfp_conf.fw_bit, self.bfp_conf.fw_dim, 1)
        else:
            gamma_ = gamma
        if self.training:
            B, C, H, W = input_.shape
            y = input_.transpose(0, 1).contiguous()  # C x B x H x W
            y = y.view(C, self.num_chunks, B * H * W // self.num_chunks)
            avg_max = y.max(-1)[0].mean(-1)  # C
            avg_min = y.min(-1)[0].mean(-1)  # C
            avg = y.view(C, -1).mean(-1)  # C
            scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
                                        0.5) / ((2 * math.log(y.size(-1))) ** 0.5)

            scale = 1 / ((avg_max - avg_min) * scale_fix + self.eps)
            with torch.no_grad():
                self.running_mean.mul_(self.momentum).add_(
                    avg * (1 - self.momentum))

                self.running_var.mul_(self.momentum).add_(
                    scale * (1 - self.momentum))
        else:
            avg = self.running_mean
            scale = self.running_var

        avg = avg.view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1)

        output = (input_ - avg) * scale
        output = output * gamma_ + beta

        if self.bfp_conf.fo:
            output = make_groups_tensor(output.clone().detach(), self.bfp_conf.fo_bit, self.bfp_conf.fo_dim, 2)

        return output"""
        #return BFPRangeBatchNorm2dFunction.apply(input, gamma, beta, self.bfp_conf, self.running_mean, self.running_var, self.momentum, self.num_chunks, self.eps, self.training)

    def extra_repr(self):
        s = ('{num_features}')
        s += ', bfp_conf=({bfp_conf})'
        if self.beta is None:
            s += ', beta=False'
        else:
            s += ', beta=True'
        return s.format(**self.__dict__)

class RangeBatchNorm2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma, beta, num_chunks=8, eps=1e-5, dtype='bfloat16'):    #input, weight, bias=None, bf_conf=None, stride=1, padding=0, dilation=1, groups=1):
        # print("= Forward:",input.shape, weight.shape, stride, padding, dilation, groups)
        # Grouping input and weight
        if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
            input = set_precision(input.clone().detach(), dtype)
        B, C, H, W = input.shape
        y = input.transpose(0, 1).contiguous()  # C x B x H x W
        y = y.view(C, num_chunks, B * H * W // num_chunks)
        avg_max = y.max(-1)[0].mean(-1)  # C
        avg_min = y.min(-1)[0].mean(-1)  # C
        avg = y.view(C, -1).mean(-1)  # C
        if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
            avg_max = set_precision(avg_max.clone().detach(), dtype)
            avg_min = set_precision(avg_min.clone().detach(), dtype)
            avg = set_precision(avg.clone().detach(), dtype)
        #scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
        #                            0.5) / ((2 * math.log(y.size(-1))) ** 0.5)
        scale_fix = 1 / ((2 * math.log(y.size(-1))) ** 0.5)
        scale = 1 / ((avg_max - avg_min) * scale_fix + eps)  
        if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
            scale = set_precision(scale.clone().detach(), dtype)
        avg = avg.view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1)
        ctx.avg = avg
        ctx.eps = eps
        ctx.scale = scale
        ctx.scale_fix = scale_fix
        ctx.num_chunks = num_chunks
        ctx.dtype = dtype

        output = (input - avg) * scale
        if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
            output = set_precision(output.clone().detach(), dtype)
        ctx.save_for_backward(input, gamma, beta, output, scale)
        output = output * gamma + beta
        if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
            output = set_precision(output.clone().detach(), dtype)

        #if bfp_conf.fo:
        #    output = make_groups_tensor(output.clone().detach(), bfp_conf.fo_bit, bfp_conf.fo_dim, 2)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors and configs
        input, gamma, beta, output, scale = ctx.saved_variables
        B, C, H, W = input.shape
        avg = ctx.avg
        scale = ctx.scale
        eps = ctx.eps
        scale_fix = ctx.scale_fix 
        num_chunks = ctx.num_chunks
        dtype = ctx.dtype
        if dtype != 'fp32':
            dtype = 'bfloat16'
        if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
            grad_output = set_precision(grad_output.clone().detach(), dtype)
        grad_input = (gamma / scale * scale) * (scale_fix - 1 / (num_chunks * B * H * W)) * grad_output

        #dL_dxi_hat = grad_output * gamma
        #dL_davg = (dL_dxi_hat * -1.0 * scale).sum(dim=(0, 2, 3), keepdim=True)
        #grad_input = dL_dxi_hat * scale + dL_davg / B
        grad_gamma = (grad_output * output).sum(dim=(0, 2, 3), keepdim=True)
        grad_beta = (grad_output).sum(dim=(0, 2, 3), keepdim=True)

        if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
            grad_input = set_precision(grad_input.clone().detach(), dtype)
            grad_gamma = set_precision(grad_gamma.clone().detach(), dtype)
            grad_beta = set_precision(grad_beta.clone().detach(), dtype)
        return grad_input, grad_gamma, grad_beta, None, None, None
        
        
# Blockfloat Batchnorm
class RangeBatchNorm2d_custom(torch.nn.Module):
    def __init__(self,
                num_features: int,
                dtype='bfloat16',
                momentum = 0.1,
                num_chunks = 8,
                eps = 1e-5,
                beta: bool = True):
        super(RangeBatchNorm2d_custom, self).__init__()

        self.num_features = num_features
        self.momentum = momentum
        self.dtype = dtype
        self.num_chunks = num_chunks
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(num_features))
        if beta:
            self.beta = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self): # -> None:
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.uniform_()
        self.beta.data.zero_()

    def forward(self, input):
        
        gamma = self.weight.view(1, self.num_features, 1, 1)
        beta = self.beta.view(1, self.num_features, 1, 1)

        return RangeBatchNorm2dFunction.apply(input, gamma, beta, self.num_chunks, self.eps, self.dtype)

    def extra_repr(self):
        s = ('{num_features}')
        if self.beta is None:
            s += ', beta=False'
        else:
            s += ', beta=True'
        return s.format(**self.__dict__)

def unsqueeze_all(t):
    # Helper function to unsqueeze all the dimensions that we reduce over
    return t[None, :, None, None]

class BFPBatchNorm2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, gamma, beta, bfp_conf, eps=1e-3):
        # Don't save keepdim'd values for backward
        if bfp_conf.fi:
            X_ = make_groups_tensor(X.clone().detach(), bfp_conf.fi_bit, bfp_conf.fi_dim, 0)
        else:
            X_ = X
        if bfp_conf.fw:
            gamma_ = make_groups_tensor(gamma.clone().detach(), bfp_conf.fw_bit, bfp_conf.fw_dim, 1)
        else:
            gamma_ = gamma

        B, C, H, W = X_.shape
        y = X_.transpose(0, 1).contiguous()  # C x B x H x W
        avg = y.view(C, -1).mean(-1)  # C
        var = y.view(C, -1).var(-1, unbiased=False)
        
        avg = avg.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        ctx.eps = eps
        ctx.avg = avg
        ctx.B = B
        ctx.var = var
        ctx.bfp_conf = bfp_conf
        output = X_ - avg #unsqueeze_all(avg)
        scale = 1 / torch.sqrt(var + eps)
        output *= scale
        if bfp_conf.fo:
            output = make_groups_tensor(output.clone().detach(), bfp_conf.fo_bit, bfp_conf.fo_dim, 2)
        ctx.save_for_backward(X_, gamma_, beta, output, scale)
        
        output = output * gamma + beta
        if bfp_conf.fo:
            output = make_groups_tensor(output.clone().detach(), bfp_conf.fo_bit, bfp_conf.fo_dim, 2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        X, gamma, beta, output, scale = ctx.saved_tensors
        B, C, H, W = X.shape
        avg = ctx.avg
        var = ctx.var
        eps = ctx.eps
        bfp_conf = ctx.bfp_conf

        if bfp_conf.bio:
            grad_output_ = make_groups_tensor(grad_output.clone().detach(), bfp_conf.bio_bit, bfp_conf.bio_dim, 10)
        else: # Apply original gradient if grad_output is not grouped
            grad_output_ = grad_output

        if bfp_conf.biw:
            if (bfp_conf.biw_bit != bfp_conf.fw_bit or bfp_conf.biw_dim != bfp_conf.fw_dim):
                gamma_ = make_groups_tensor(gamma.clone().detach(), bfp_conf.biw_bit, bfp_conf.biw_dim, 11)
            else:
                gamma_ = gamma
        else:
            gamma_ = gamma

        dL_dxi_hat = grad_output_ * gamma_
        dL_dvar = (dL_dxi_hat * (X - avg) * -0.5 * scale * scale * scale).sum(dim=(0, 2, 3), keepdim=True)
        
        dL_davg = (dL_dxi_hat * -1.0 * scale).sum(dim=(0, 2, 3), keepdim=True) + dL_dvar * (-2.0 * (X - avg)).sum(dim=(0, 2, 3), keepdim=True) / (B*H*W) 
        dL_dxi = dL_dxi_hat * scale + dL_dvar * 2.0 * (X - avg) / (B*H*W)  + dL_davg / (B*H*W)
        dL_dgamma = (grad_output * output).sum(dim=(0, 2, 3), keepdim=True)
        dL_dbeta = (grad_output).sum(dim=(0, 2, 3), keepdim=True)

        if bfp_conf.big and dL_dxi != None:
            dL_dxi = make_groups_tensor(dL_dxi.clone().detach(), bfp_conf.big_bit,bfp_conf.big_dim, 12)
        if bfp_conf.bwg and dL_dgamma != None:
            dL_dgamma = make_groups_tensor(dL_dgamma.clone().detach(), bfp_conf.bwg_bit, bfp_conf.bwg_dim, 22)

        return dL_dxi, dL_dgamma, dL_dbeta, None, None

class BFPBatchNorm2d_custom(nn.Module):
    def __init__(self, num_features, bfp_conf=BFPConf, momentum=0.1, eps=1e-5):
        super(BFPBatchNorm2d_custom, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.bfp_conf = bfp_conf
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()
        self.eps = eps

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, inp):
        gamma = self.weight.view(1, self.num_features, 1, 1)
        beta = self.bias.view(1, self.num_features, 1, 1)
        return BFPBatchNorm2dFunction.apply(inp, gamma, beta, self.bfp_conf, self.eps)
    
    def extra_repr(self):
        # From /torch/nn/modules/conv.py
        s = ('{num_features}')
        s += ', bfp_conf=({bfp_conf})'
        return s.format(**self.__dict__)

class BatchNorm2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, gamma, beta, eps=1e-3, dtype='fp8'):
        # Don't save keepdim'd values for backward
        if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
            X = set_precision(X.clone().detach(), dtype)
            gamma = set_precision(gamma.clone().detach(), dtype)
            dtype_nacc = dtype
            if '+' in dtype:
                dtype_nacc = dtype.split('+')[0]
            B, C, H, W = X.shape
            y = X.transpose(0, 1).contiguous()  # C x B x H x W
            sum = y.view(C, -1).sum(-1)
            n = y.view(C, -1).size()[1]
            avg = sum / n
            avg = set_precision(avg.clone().detach(), dtype_nacc)
            var = (y.view(C, -1) - avg.view(C, -1)) ** 2
            var = set_precision(var.clone().detach(), dtype_nacc)
            var = var.sum(-1)
            var = set_precision(var.clone().detach(), dtype)
            var = var / n
            var = set_precision(var.clone().detach(), dtype_nacc)

            avg = avg.view(1, -1, 1, 1)
            var = var.view(1, -1, 1, 1)
            ctx.eps = eps
            ctx.avg = avg
            ctx.B = B
            ctx.var = var
            ctx.dtype = dtype
            output = X - avg #unsqueeze_all(avg)
            scale = 1 / torch.sqrt(var + eps)
            output *= scale
            output = set_precision(output.clone().detach(), dtype_nacc)
            ctx.save_for_backward(X, gamma, beta, output, scale)
            
            output = output * gamma
            output = set_precision(output.clone().detach(), dtype_nacc)
            output += beta
            output = set_precision(output.clone().detach(), dtype)
        else:
            B, C, H, W = X.shape
            y = X.transpose(0, 1).contiguous()  # C x B x H x W
            avg = y.view(C, -1).mean(-1)
            var = y.view(C, -1).var(-1, unbiased=False)

            avg = avg.view(1, -1, 1, 1)
            var = var.view(1, -1, 1, 1)
            ctx.eps = eps
            ctx.avg = avg
            ctx.B = B
            ctx.var = var
            ctx.dtype = dtype
            output = X - avg #unsqueeze_all(avg)
            scale = 1 / torch.sqrt(var + eps)
            output *= scale
            ctx.save_for_backward(X, gamma, beta, output, scale)
            
            output = output * gamma + beta
        return output

    @staticmethod
    def backward(ctx, grad_output):
        X, gamma, beta, output, scale = ctx.saved_tensors
        B, C, H, W = X.shape
        avg = ctx.avg
        var = ctx.var
        eps = ctx.eps
        dtype = ctx.dtype
        dtype_nacc = dtype
        if dtype != 'fp32':
            dtype = 'bfloat16'
        if '+' in dtype:
            dtype_nacc = dtype.split('+')[0]
        if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
            dL_dxi_hat = grad_output * gamma
            dL_dxi_hat = set_precision(dL_dxi_hat.clone().detach(), dtype_nacc)
            dL_dvar = dL_dxi_hat * (X - avg) * -0.5 * scale * scale * scale
            dL_dvar = set_precision(dL_dvar.clone().detach(), dtype_nacc)
            dL_dvar = dL_dvar.sum(dim=(0, 2, 3), keepdim=True)
            dL_dvar = set_precision(dL_dvar.clone().detach(), dtype)
            dL_dvar_ori = (dL_dxi_hat * (X - avg) * -0.5 * scale * scale * scale).sum(dim=(0, 2, 3), keepdim=True)
            #print('dL_dvar', dL_dvar - dL_dvar_ori)
            a = set_precision((dL_dxi_hat * -1.0 * scale).clone().detach(), dtype_nacc)
            a_tmp = set_precision((a.sum(dim=(0, 2, 3), keepdim=True)).clone().detach(), dtype)
            print('a_tmp', a_tmp)
            b = dL_dxi_hat * -1.0 * scale
            b = b.sum(dim=(0, 2, 3), keepdim=True)
            print('b', b)
            #print(a_tmp - b)
            dL_davg1 = set_precision(((set_precision((dL_dxi_hat * -1.0 * scale).clone().detach(), dtype_nacc)).sum(dim=(0, 2, 3), keepdim=True)).clone().detach(), dtype)
            dL_davg1_ori = (dL_dxi_hat * -1.0 * scale).sum(dim=(0, 2, 3), keepdim=True)
            #print('dL_davg1', dL_davg1 - dL_davg1_ori)
            dL_davg2 = set_precision((dL_dvar * set_precision((-2.0 * (X - avg)).sum(dim=(0, 2, 3), keepdim=True).clone().detach(), dtype) / (B*H*W)).clone().detach(), dtype)
            dL_davg = set_precision((dL_davg1 + dL_davg2).clone().detach(), dtype)
            dL_davg_ori = (dL_dxi_hat * -1.0 * scale).sum(dim=(0, 2, 3), keepdim=True) + dL_dvar * (-2.0 * (X - avg)).sum(dim=(0, 2, 3), keepdim=True) / (B*H*W)

            #print('dL_davg', dL_davg - dL_davg_ori)
            dL_davg = set_precision(dL_davg.clone().detach(), dtype_nacc)
            dL_dxi = dL_dxi_hat * scale + dL_dvar * 2.0 * (X - avg) / (B*H*W)  + dL_davg / (B*H*W)
            dL_dgamma = (grad_output * output).sum(dim=(0, 2, 3), keepdim=True)
            dL_dbeta = (grad_output).sum(dim=(0, 2, 3), keepdim=True)
            if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
                dL_dxi = set_precision(dL_dxi.clone().detach(), dtype)
                dL_dgamma = set_precision(dL_dgamma.clone().detach(), dtype)
                dL_dbeta = set_precision(dL_dbeta.clone().detach(), dtype)
        if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
            if '+' in dtype:
                dtype_nacc = dtype.split('+')[0]
            grad_output = set_precision(grad_output.clone().detach(), dtype)
        if '+' in dtype:
            dL_dxi_hat = grad_output * gamma
            if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
                dL_dxi_hat = set_precision(dL_dxi_hat.clone().detach(), dtype)
        else:
            dL_dxi_hat = grad_output * gamma
            if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
                dL_dxi_hat = set_precision(dL_dxi_hat.clone().detach(), dtype)
            dL_dvar = (dL_dxi_hat * (X - avg) * -0.5 * scale * scale * scale).sum(dim=(0, 2, 3), keepdim=True)
            if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
                dL_dvar = set_precision(dL_dvar.clone().detach(), dtype)
            dL_davg = (dL_dxi_hat * -1.0 * scale).sum(dim=(0, 2, 3), keepdim=True) + dL_dvar * (-2.0 * (X - avg)).sum(dim=(0, 2, 3), keepdim=True) / (B*H*W)
            if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
                dL_davg = set_precision(dL_davg.clone().detach(), dtype)
            dL_dxi = dL_dxi_hat * scale + dL_dvar * 2.0 * (X - avg) / (B*H*W)  + dL_davg / (B*H*W)
            dL_dgamma = (grad_output * output).sum(dim=(0, 2, 3), keepdim=True)
            dL_dbeta = (grad_output).sum(dim=(0, 2, 3), keepdim=True)
            if dtype == 'bfloat16' or dtype == 'fp16' or dtype == 'fp16+4' or dtype == 'fp16+8' or dtype == 'fp8' or dtype == 'fp8+4' or dtype == 'fp8+8':
                dL_dxi = set_precision(dL_dxi.clone().detach(), dtype)
                dL_dgamma = set_precision(dL_dgamma.clone().detach(), dtype)
                dL_dbeta = set_precision(dL_dbeta.clone().detach(), dtype)
        return dL_dxi, dL_dgamma, dL_dbeta, None, None

class BatchNorm2d_custom(nn.Module):
    def __init__(self, num_features, dtype='bfloat16', momentum=0.1, eps=1e-5):
        super(BatchNorm2d_custom, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.dtype = dtype
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()
        self.eps = eps

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, inp):
        gamma = self.weight.view(1, self.num_features, 1, 1)
        beta = self.bias.view(1, self.num_features, 1, 1)
        return BatchNorm2dFunction.apply(inp, gamma, beta, self.eps, self.dtype)

    def extra_repr(self):
        # From /torch/nn/modules/conv.py
        s = ('{num_features}')
        s += ', {dtype}'
        return s.format(**self.__dict__)