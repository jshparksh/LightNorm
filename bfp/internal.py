import torch
import numpy as np
import ctypes

from conf import FLAGS, CUDA_THREADSPERBLOCK


fp32_mask = [0,
    0x00400000, 0x00600000, 0x00700000, 0x00780000,
    0x007c0000, 0x007e0000, 0x007f0000, 0x007f8000,
    0x007fc000, 0x007fe000, 0x007ff000, 0x007ff800,
    0x007ffc00, 0x007ffe00, 0x007fff00, 0x007fff80,
    0x007fffc0, 0x007fffe0, 0x007ffff0, 0x007ffff8, 0x007fffff]

fp64_mask = [0,
    0x0040000000000000, 0x0060000000000000, 0x0070000000000000, 0x0078000000000000,
    0x007c000000000000, 0x007e000000000000, 0x007f000000000000, 0x007f800000000000,
    0x007fc00000000000, 0x007fe00000000000, 0x007ff00000000000, 0x007ff80000000000,
    0x007ffc0000000000, 0x007ffe0000000000, 0x007fff0000000000, 0x007fff8000000000]

from numba import jit, cuda
import numba

@cuda.jit
# TODO : make another function to just grouping tensor...?
def make_groups_3d_internal(v, dim, bs, gs, group_mantissa):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x 
    
    idx0o = (idx // (bs[2] * bs[1])) % bs[0] * gs[0]
    idx1o = (idx // (bs[2])) % bs[1] * gs[1]
    idx2o = idx % bs[2] * gs[2]

    M = 0
    for idx0 in range(idx0o, idx0o + gs[0]):
        if idx0 >= dim[0]:
            break
        for idx1 in range(idx1o, idx1o + gs[1]):
            if idx1 >= dim[1]:
                break
            for idx2 in range(idx2o, idx2o + gs[2]):
                if idx2 >= dim[2]:
                    break
                e = (v[idx0,idx1,idx2] & 0x7f800000 ) >> 23
                if e < 0:
                    e = -e
                if M < e:
                    M = e

    # Replace that area
    for idx0 in range(idx0o, idx0o + gs[0]):
        if idx0 >= dim[0]:
            break
        for idx1 in range(idx1o, idx1o + gs[1]):
            if idx1 >= dim[1]:
                break
            for idx2 in range(idx2o, idx2o + gs[2]):
                if idx2 >= dim[2]:
                    break
                e = (v[idx0,idx1,idx2] & 0x7f800000 ) >> 23
                k = group_mantissa - M + e - 1
                if 0 <= k:
                    v[idx0,idx1,idx2] = v[idx0,idx1,idx2] & (0xffffffff << (23 - k))
                else:
                    v[idx0,idx1,idx2] = 0

@cuda.jit
# TODO : make another function to just grouping tensor...?
def make_groups_4d_internal(v, dim, bs, gs, group_mantissa):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x 
    
    idx0o = (idx // (bs[3] * bs[2] * bs[1])) % bs[0] * gs[0]
    idx1o = (idx // (bs[3] * bs[2])) % bs[1] * gs[1]
    idx2o = (idx // bs[3]) % bs[2] * gs[2]
    idx3o = idx % bs[3] * gs[3]

    M = 0
    for idx0 in range(idx0o, idx0o + gs[0]):
        if idx0 >= dim[0]:
            break
        for idx1 in range(idx1o, idx1o + gs[1]):
            if idx1 >= dim[1]:
                break
            for idx2 in range(idx2o, idx2o + gs[2]):
                if idx2 >= dim[2]:
                    break
                for idx3 in range(idx3o, idx3o + gs[3]):
                    if idx3 >= dim[3]:
                        break
                    e = (v[idx0,idx1,idx2,idx3] & 0x7f800000 ) >> 23
                    if e < 0:
                        e = -e
                    if M < e:
                        M = e

    # Replace that area
    for idx0 in range(idx0o, idx0o + gs[0]):
        if idx0 >= dim[0]:
            break
        for idx1 in range(idx1o, idx1o + gs[1]):
            if idx1 >= dim[1]:
                break
            for idx2 in range(idx2o, idx2o + gs[2]):
                if idx2 >= dim[2]:
                    break
                for idx3 in range(idx3o, idx3o + gs[3]):
                    if idx3 >= dim[3]:
                        break
                    e = (v[idx0,idx1,idx2,idx3] & 0x7f800000 ) >> 23
                    k = group_mantissa - M + e - 1
                    if 0 <= k:
                        v[idx0,idx1,idx2,idx3] = v[idx0,idx1,idx2,idx3] & (0xffffffff << (23 - k))
                    else:
                        v[idx0,idx1,idx2,idx3] = 0

# make_group_tensor : Group values as same exponent bits, which shifts mantissa
def make_groups_tensor(inp, group_mantissa, group_dim, type = -1):
    if group_dim == 1 or group_dim == (1):
        group_dim = []
        for i in len(inp.size()):
            group_dim.append(1)
        group_dim = tuple(group_dim)
    
    """
    s = "==%s============================\n"%(str(inpsize))
    for i in range(0, 8):
        for j in range(0, 3):
            s += "%2.3f\t"%inp[i,j,0,0]
        s += "\t"
    s += "\n"
    # """

    # inp = cuda.to_device(inp)
    inp_ = inp.view(torch.int32)
    if len(inp.size()) == 4:
        bs = ((inp.size()[0]-1)//group_dim[0]+1, (inp.size()[1]-1)//group_dim[1]+1, (inp.size()[2]-1)//group_dim[2]+1, (inp.size()[3]-1)//group_dim[3]+1)
        blockspergrid = (inp.size()[0]*inp.size()[1]*inp.size()[2]*inp.size()[3] +  (CUDA_THREADSPERBLOCK - 1)) // CUDA_THREADSPERBLOCK
        inpsize = (inp.size()[0], inp.size()[1], inp.size()[2], inp.size()[3])

        make_groups_4d_internal[blockspergrid, CUDA_THREADSPERBLOCK](inp_, inpsize, bs, group_dim, group_mantissa)
    elif len(inp.size()) == 3:
        bs = ((inp.size()[0]-1)//group_dim[0]+1, (inp.size()[1]-1)//group_dim[1]+1, (inp.size()[2]-1)//group_dim[2]+1)
        blockspergrid = (inp.size()[0]*inp.size()[1]*inp.size()[2] +  (CUDA_THREADSPERBLOCK - 1)) // CUDA_THREADSPERBLOCK
        inpsize = (inp.size()[0], inp.size()[1], inp.size()[2])

        make_groups_3d_internal[blockspergrid, CUDA_THREADSPERBLOCK](inp_, inpsize, bs, group_dim, group_mantissa)
    else:
        # Do nothing
        print("Tensor not supported, didn't do anything")
        return inp
    # inp = inp.copy_to_host()

    """
    for i in range(0, 8):
        for j in range(0, 3):
            s += "%2.3f\t"%v[i,j,0,0]
        s += "\t"
    s += "\n"
    # """
    # s = "%d / %d"%(np.count_nonzero(v == 0), inpsize[0]*inpsize[1]*inpsize[2]*inpsize[3])
    # print(s)
    # return torch.from_numpy(v).cuda()
    return inp
