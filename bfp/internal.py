"""
    This code is part of the BFPSim (https://github.com/ids-Lab-DGIST/BFPSim)
<<<<<<< HEAD
    Seunghyun Lee (R3C0D3r) from IDSLab, DGIST
    coder@dgist.ac.kr
=======

    Seunghyun Lee (R3C0D3r) from IDSLab, DGIST
    coder@dgist.ac.kr

>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
    License: CC BY 4.0
"""

import torch
import numpy as np
import ctypes

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

<<<<<<< HEAD
# from conf import FLAGS
# from utils.tensorAnalyze import analyzeObject
@cuda.jit
# TODO : make another function to just grouping tensor...?
def make_groups_1d_internal(v, dim, bs, gs, group_mantissa):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x 
    
    idx0o = (idx // bs) * gs[0]

    # Find the max and second max exponent from each group
    M_max = 0
    M_second_max = 0
    for idx0 in range(idx0o, idx0o + gs[0]):
        if idx0 >= dim:
            break
        e = (v[idx0] & 0x7f800000 ) >> 23
        if M_second_max < e:
            if M_max < e:
                M_max, M_second_max = e, M_max
            else:
                M_second_max = e
    if M_max == 0:
        return
    # Replace that area
    for idx0 in range(idx0o, idx0o + gs[0]):
        if idx0 >= dim:
            break
        e = (v[idx0] & 0x7f800000 ) >> 23
        if e == M_max:
            v[idx0] = v[idx0]
        else:
            k = group_mantissa - M_second_max + e - 1
            if 0 <= k:
                v[idx0] = v[idx0] & (0xffffffff << (23 - k))
            else:
                v[idx0] = 0
    cuda.syncthreads()

@cuda.jit
=======
@cuda.jit
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
def make_groups_2d_internal(v, dim, bs, gs, group_mantissa):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x 
    
    idx0o = (idx // bs[1]) * gs[0]
    idx1o = idx % bs[1] * gs[1]

<<<<<<< HEAD
     # Find the max and second max exponent from each group
    M_max = 0
    M_second_max = 0
=======
    # Find the max exponent from each group
    M = 0
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
    for idx0 in range(idx0o, idx0o + gs[0]):
        if idx0 >= dim[0]:
            break
        for idx1 in range(idx1o, idx1o + gs[1]):
            if idx1 >= dim[1]:
                break
            e = (v[idx0,idx1] & 0x7f800000 ) >> 23
            if M_second_max < e:
                if M_max < e:
                    M_max, M_second_max = e, M_max
                else:
                    M_second_max = e
    if M_max == 0:
        return
    # Remove each mantissa to desired values
    for idx0 in range(idx0o, idx0o + gs[0]):
        if idx0 >= dim[0]:
            break
        for idx1 in range(idx1o, idx1o + gs[1]):
            if idx1 >= dim[1]:
                break
            e = (v[idx0,idx1] & 0x7f800000 ) >> 23
            if e == M_max:
                v[idx0,idx1] = v[idx0,idx1]
            else:
                k = group_mantissa - M_second_max + e - 1
                if 0 <= k:
                    v[idx0,idx1] = v[idx0,idx1] & (0xffffffff << (23 - k))
                else:
                    v[idx0,idx1] = 0
    cuda.syncthreads()

@cuda.jit
def make_groups_3d_internal(v, dim, bs, gs, group_mantissa):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x 
    
    idx0o = (idx // (bs[2] * bs[1])) * gs[0]
    idx1o = (idx // bs[2]) % bs[1] * gs[1]
    idx2o = idx % bs[2] * gs[2]

<<<<<<< HEAD
    # Find the max and second max exponent from each group
    M_max = 0
    M_second_max = 0
=======
    # Find the max exponent from each group
    M = 0
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
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
                if M_second_max < e:
                    if M_max < e:
                        M_max, M_second_max = e, M_max
                    else:
                        M_second_max = e
    if M_max == 0:
        return
    # Remove each mantissa to desired values
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
                if e == M_max:
                    v[idx0,idx1,idx2] = v[idx0,idx1,idx2]
                else:
                    k = group_mantissa - M_second_max + e - 1
                    if 0 <= k:
                        v[idx0,idx1,idx2] = v[idx0,idx1,idx2] & (0xffffffff << (23 - k))
                    else:
                        v[idx0,idx1,idx2] = 0
    cuda.syncthreads()

@cuda.jit
# TODO : make another function to just grouping tensor...?
def make_groups_4d_internal(v, dim, bs, gs, group_mantissa):
<<<<<<< HEAD
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x
=======
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x     
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
    idx0o = (idx // (bs[3] * bs[2] * bs[1])) * gs[0]
    idx1o = (idx // (bs[3] * bs[2])) % bs[1] * gs[1]
    idx2o = (idx // bs[3]) % bs[2] * gs[2]
    idx3o = idx % bs[3] * gs[3]

<<<<<<< HEAD
    # Find the max and second max exponent from each group
    M_max = 0
    M_second_max = 0
=======
    # Find the max exponent from each group
    M = 0
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
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
                    if M_second_max < e:
                        if M_max < e:
                            M_max, M_second_max = e, M_max
                        else:
                            M_second_max = e
    if M_max == 0:
        return
    # Remove each mantissa to desired values
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
                    if e == M_max:
                        v[idx0,idx1,idx2,idx3] = v[idx0,idx1,idx2,idx3]
                    else:
                        k = group_mantissa - M_second_max + e - 1
                        if 0 <= k:
                            v[idx0,idx1,idx2,idx3] = v[idx0,idx1,idx2,idx3] & (0xffffffff << (23 - k))
                        else:
                            v[idx0,idx1,idx2,idx3] = 0
    cuda.syncthreads()

# make_group_tensor : Group values as same exponent bits, which shifts mantissa
def make_groups_tensor(inp, group_mantissa, group_dim, type = -1):
<<<<<<< HEAD
    # Make true to ZSE analyze, temporal disabled
    # if FLAGS.ZSE:
    #     analyzeObject.AddData(inp.clone().detach(), group_mantissa, group_dim, type)

    # Set pointer of tensor as int, easier to manipulate
    print(inp.type())
=======

    # Set pointer of tensor as int, easier to manipulate
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
    inp_ = inp.view(torch.int32)
    ins = np.array(inp.size())
    if len(ins) == 4: # If the tensor size is 4d
        # Choose Ideal thread size
        threads = (ins[0]*ins[1]*ins[2]*ins[3]) // (group_dim[0]*group_dim[1]*group_dim[2]*group_dim[3])
        threads = threads + (32 - threads % 32)
        if threads > 1024:
            threads = 1024
        threads = int(threads)

        # Set blockspergrid and call internal function
        bs = ((ins[0]-1)//group_dim[0]+1, (ins[1]-1)//group_dim[1]+1, (ins[2]-1)//group_dim[2]+1, (ins[3]-1)//group_dim[3]+1)
        blockspergrid = (ins[0]*ins[1]*ins[2]*ins[3] +  (threads - 1)) // threads
        inpsize = (ins[0], ins[1], ins[2], ins[3])
        make_groups_4d_internal[blockspergrid, threads](inp_, inpsize, bs, group_dim, group_mantissa)
    elif len(ins) == 3: # If the tensor size is 3d
        # Choose Ideal thread size
        threads = (ins[0]*ins[1]*ins[2]) // (group_dim[0]*group_dim[1]*group_dim[2])
        threads = threads + (32 - threads % 32)
        if threads > 1024:
            threads = 1024
        threads = int(threads)

        # Set blockspergrid and call internal function
        bs = ((ins[0]-1)//group_dim[0]+1, (ins[1]-1)//group_dim[1]+1, (ins[2]-1)//group_dim[2]+1)
        blockspergrid = (ins[0]*ins[1]*ins[2] + (threads - 1)) // threads
        inpsize = (ins[0], ins[1], ins[2])
        make_groups_3d_internal[blockspergrid, threads](inp_, inpsize, bs, group_dim, group_mantissa)
    elif len(ins) == 2:
        # Choose Ideal thread size
        threads = (ins[0]*ins[1]) // (group_dim[0]*group_dim[1])
        threads = threads + (32 - threads % 32)
        if threads > 1024:
            threads = 1024
        threads = int(threads)
        
        # Set blockspergrid and call internal function
        bs = ((ins[0]-1)//group_dim[0]+1, (ins[1]-1)//group_dim[1]+1)
        blockspergrid = (ins[0]*ins[1] + (threads - 1)) // threads
        inpsize = (ins[0], ins[1])
        make_groups_2d_internal[blockspergrid, threads](inp_, inpsize, bs, group_dim, group_mantissa)
<<<<<<< HEAD
    elif len(ins) == 1:
        # Choose Ideal thread size
        threads = (ins[0]) // (group_dim[0])
        threads = threads + (32 - threads % 32)
        if threads > 1024:
            threads = 1024
        threads = int(threads)

        bs = ((ins[0]-1)//group_dim[0]+1)
        blockspergrid = (ins[0] + (threads - 1)) // threads
        inpsize = (ins[0])
        make_groups_1d_internal[blockspergrid, threads](inp_, inpsize, bs, group_dim, group_mantissa)
=======
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
    else: # Tensor dimension is not supported
        inpsize = (ins[0], ins[1], ins[2], ins[3])
        print("make_groups_tensor ERROR: Tensor dimension not supported %s"%(str(inpsize)))
        return inp

    return inp

<<<<<<< HEAD
@cuda.jit
# TODO : make another function to just grouping tensor...?
def set_bfloat16_4d_internal(v, dim):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x
    i0 = (idx // (dim[3]*dim[2]*dim[1])) % dim[0]
    i1 = (idx // (dim[3]*dim[2])) % dim[1]
    i2 = (idx // (dim[3])) % dim[2]
    i3 = idx % dim[3]
    
    if i0 >= dim[0] or i1 >= dim[1] or i2 >= dim[2] or i3 >= dim[3]:
        return
    v[i0,i1,i2,i3] = v[i0,i1,i2,i3] & (0xffff0000)

@cuda.jit
# TODO : make another function to just grouping tensor...?
def set_bfloat16_4_4d_internal(v, dim, precision):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x
    i0 = (idx // (dim[3]*dim[2]*dim[1])) % dim[0]
    i1 = (idx // (dim[3]*dim[2])) % dim[1]
    i2 = (idx // (dim[3])) % dim[2]
    i3 = idx % dim[3]
    
    if i0 >= dim[0] or i1 >= dim[1] or i2 >= dim[2] or i3 >= dim[3]:
        return
    eb = v[i0,i1,i2,i3] & 0x7f800000
    mb = (0xffffffff << (23 - precision)) & 0x007fffff
    v[i0,i1,i2,i3] = v[i0,i1,i2,i3] & ( 0x80000000 | eb | mb )

@cuda.jit
# TODO : make another function to just grouping tensor...?
def set_fp8_4d_internal(v, dim, precision):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x
    i0 = (idx // (dim[3]*dim[2]*dim[1])) % dim[0]
    i1 = (idx // (dim[3]*dim[2])) % dim[1]
    i2 = (idx // (dim[3])) % dim[2]
    i3 = idx % dim[3]
    
    if i0 >= dim[0] or i1 >= dim[1] or i2 >= dim[2] or i3 >= dim[3]:
        return
    e_ = ( v[i0,i1,i2,i3] >> 23 ) & 0xff
    if e_ < 127 - 14:
        eb = ( 127 - 14 ) << 23
    elif e_ > 127 + 15:
        eb = ( 127 + 15 ) << 23
    else:
        eb = v[i0,i1,i2,i3] & 0x7f800000
    mb = (0xffffffff << (23 - precision)) & 0x007fffff
    v[i0,i1,i2,i3] = v[i0,i1,i2,i3] & ( 0x80000000 | eb | mb )

@cuda.jit
# TODO : make another function to just grouping tensor...?
def set_fp8_143_4d_internal(v, dim, precision):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x
    i0 = (idx // (dim[3]*dim[2]*dim[1])) % dim[0]
    i1 = (idx // (dim[3]*dim[2])) % dim[1]
    i2 = (idx // (dim[3])) % dim[2]
    i3 = idx % dim[3]
    
    if i0 >= dim[0] or i1 >= dim[1] or i2 >= dim[2] or i3 >= dim[3]:
        return
    e_ = ( v[i0,i1,i2,i3] >> 23 ) & 0xff
    if e_ < 127 - 6:
        eb = ( 127 - 6 ) << 23
    elif e_ > 127 + 7:
        eb = ( 127 + 7 ) << 23
    else:
        eb = v[i0,i1,i2,i3] & 0x7f800000
    mb = (0xffffffff << (23 - precision)) & 0x007fffff
    v[i0,i1,i2,i3] = v[i0,i1,i2,i3] & ( 0x80000000 | eb | mb )
    
@cuda.jit
# TODO : make another function to just grouping tensor...?
def set_fp16_4d_internal(v, dim, precision):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x
    i0 = (idx // (dim[3]*dim[2]*dim[1])) % dim[0]
    i1 = (idx // (dim[3]*dim[2])) % dim[1]
    i2 = (idx // (dim[3])) % dim[2]
    i3 = idx % dim[3]
    
    if i0 >= dim[0] or i1 >= dim[1] or i2 >= dim[2] or i3 >= dim[3]:
        return

    e_ = ( v[i0,i1,i2,i3] >> 23 ) & 0xff
    if e_ < 127 - 14:
        eb = ( 127 - 14 ) << 23
    elif e_ > 127 + 15:
        eb = ( 127 + 15 ) << 23
    else:
        eb = v[i0,i1,i2,i3] & 0x7f800000
    mb = (0xffffffff << (23 - precision)) & 0x007fffff
    v[i0,i1,i2,i3] = v[i0,i1,i2,i3] & ( 0x80000000 | eb | mb )

@cuda.jit
# TODO : make another function to just grouping tensor...?
def set_fp10_154_4d_internal(v, dim, precision):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x
    i0 = (idx // (dim[3]*dim[2]*dim[1])) % dim[0]
    i1 = (idx // (dim[3]*dim[2])) % dim[1]
    i2 = (idx // (dim[3])) % dim[2]
    i3 = idx % dim[3]
    
    if i0 >= dim[0] or i1 >= dim[1] or i2 >= dim[2] or i3 >= dim[3]:
        return

    e_ = ( v[i0,i1,i2,i3] >> 23 ) & 0xff
    if e_ < 127 - 14:
        eb = ( 127 - 14 ) << 23
    elif e_ > 127 + 15:
        eb = ( 127 + 15 ) << 23
    else:
        eb = v[i0,i1,i2,i3] & 0x7f800000
    mb = (0xffffffff << (23 - precision)) & 0x007fffff
    v[i0,i1,i2,i3] = v[i0,i1,i2,i3] & ( 0x80000000 | eb | mb )

@cuda.jit
# TODO : make another function to just grouping tensor...?
def set_fp10_163_4d_internal(v, dim, precision):
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x
    i0 = (idx // (dim[3]*dim[2]*dim[1])) % dim[0]
    i1 = (idx // (dim[3]*dim[2])) % dim[1]
    i2 = (idx // (dim[3])) % dim[2]
    i3 = idx % dim[3]
    
    if i0 >= dim[0] or i1 >= dim[1] or i2 >= dim[2] or i3 >= dim[3]:
        return

    e_ = ( v[i0,i1,i2,i3] >> 23 ) & 0xff
    if e_ < 127 - 30:
        eb = ( 127 - 30 ) << 23
    elif e_ > 127 + 31:
        eb = ( 127 + 31 ) << 23
    else:
        eb = v[i0,i1,i2,i3] & 0x7f800000
    mb = (0xffffffff << (23 - precision)) & 0x007fffff
    v[i0,i1,i2,i3] = v[i0,i1,i2,i3] & ( 0x80000000 | eb | mb )

# set_precision
def set_precision(inp, dtype):
    inp_ = inp.view(torch.int32)
    ins = np.array(inp.size())
    if len(ins) == 4:
        threads = np.prod(inp.size())
        threads = (threads + 31) % 32
        if threads > 1024:
            threads = 1024
        threads = int(threads)
        inpsize = (ins[0], ins[1], ins[2], ins[3])
        blockspergrid = (np.prod(inp.size()) + threads - 1) // threads
        if dtype == 'bfloat16':
            set_bfloat16_4d_internal[blockspergrid, threads](inp_, inpsize)
        if dtype == 'bfloat16+4':
            set_bfloat16_4_4d_internal[blockspergrid, threads](inp_, inpsize, 11)
        if dtype == 'fp16':
            set_fp16_4d_internal[blockspergrid, threads](inp_, inpsize, 10)
        if dtype == 'fp16+4':
            set_fp16_4d_internal[blockspergrid, threads](inp_, inpsize, 14)
        if dtype == 'fp16+8':
            set_fp16_4d_internal[blockspergrid, threads](inp_, inpsize, 18)
        if dtype == 'fp8':
            set_fp8_4d_internal[blockspergrid, threads](inp_, inpsize, 2)
        if dtype == 'fp8+4':
            set_fp8_4d_internal[blockspergrid, threads](inp_, inpsize, 6)
        if dtype == 'fp8+8':
            set_fp8_4d_internal[blockspergrid, threads](inp_, inpsize, 10)
        if dtype == 'fp8+8_143':
            set_fp8_143_4d_internal[blockspergrid, threads](inp_, inpsize, 11)
        if dtype == 'fp10_154':
            set_fp10_154_4d_internal[blockspergrid, threads](inp_, inpsize, 4)
        if dtype == 'fp10_163':
            set_fp10_163_4d_internal[blockspergrid, threads](inp_, inpsize, 3)
        if dtype == 'fp10_154+8':
            set_fp10_154_4d_internal[blockspergrid, threads](inp_, inpsize, 12)
        if dtype == 'fp10_163+8':
            set_fp10_163_4d_internal[blockspergrid, threads](inp_, inpsize, 11)
        if dtype == 'fp12_156':
            set_fp10_154_4d_internal[blockspergrid, threads](inp_, inpsize, 6)
        if dtype == 'fp12_165':
            set_fp10_163_4d_internal[blockspergrid, threads](inp_, inpsize, 5)
        if dtype == 'fp15_168':
            set_fp10_163_4d_internal[blockspergrid, threads](inp_, inpsize, 8)
        #set_precision_4d_internal[blockspergrid, threads](inp_, inpsize, dtype)
    return inp
=======
# Linear Backward Gradient Computation
@cuda.jit
def gradient_linear_weight_internal_3d(o, a, b):
    idx = ( cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x ) % a.shape[2]
    idy = ( cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x ) // a.shape[2]

    if idx >= a.shape[2] or idy >= b.shape[2]:
        return
    # Concat calculated data to dimension of (0,1)
    tmp = 0
    for ix in range(a.shape[0]):
        for iy in range(a.shape[1]):
           tmp += a[ix,iy,idx] * b[ix,iy,idy]
    o[idx,idy] = tmp
    # cuda.syncthreads()

# gradient_linear_weight_3d : (b*n*i)*(b*n*o) = (i*o)
def gradient_linear_weight_3d(grad_output, input, weight_shape):
    # Create grad_weight to store data
    grad_weight = torch.empty((grad_output.shape[2], input.shape[2]),dtype=torch.float).cuda()
    # Handling error
    if grad_output.shape[0] != input.shape[0] or grad_output.shape[1] != input.shape[1]:
        print("gradient_linear_weight ERROR: Tensor dimension not match. [b*n*i] x [b*n*o] = [i*o], %s x %s"%(grad_output.shape, input.shape))
        return None
    if grad_output.is_cuda == False:
        print("gradient_linear_weight ERROR: grad_output is not in cuda")
        return None
    if input.is_cuda == False:
        print("gradient_linear_weight ERROR: input is not in cuda")
        return None
    # Define threads and blockspergrid (Optimizeable, I think)
    threads = grad_output.shape[2] * input.shape[2]
    threads = threads + (32 - threads % 32)
    if threads > 1024:
        threads = 1024
    threads = int(threads)
    blockspergrid = (grad_output.shape[2] * input.shape[2] + (threads - 1)) // threads
    # Call internal function to calcaluate actually
    gradient_linear_weight_internal_3d[blockspergrid, threads](grad_weight, grad_output, input)

    return grad_weight

@cuda.jit
def gradient_linear_weight_internal_2d(o, a, b):
    idx = ( cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x ) % a.shape[1]
    idy = ( cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x ) // a.shape[1]

    if idx >= a.shape[1] or idy >= b.shape[1]:
        return
    # Concat calculated data to dimension of (0,1)
    tmp = 0
    for ix in range(a.shape[0]):
        tmp += a[ix,idx] * b[ix,idy]
    o[idx,idy] = tmp
    # cuda.syncthreads()

# gradient_linear_weight_2d : (b*i)*(b*o) = (i*o)
def gradient_linear_weight_2d(grad_output, input, weight_shape):
    # Create grad_weight to store data
    grad_weight = torch.empty((grad_output.shape[1], input.shape[1]),dtype=torch.float).cuda()
    # Handling error
    if grad_output.is_cuda == False:
        print("gradient_linear_weight ERROR: grad_output is not in cuda")
        return None
    if input.is_cuda == False:
        print("gradient_linear_weight ERROR: input is not in cuda")
        return None
    # Define threads and blockspergrid (Optimizeable, I think)
    threads = grad_output.shape[1] * input.shape[1]
    threads = threads + (32 - threads % 32)
    if threads > 1024:
        threads = 1024
    threads = int(threads)
    blockspergrid = (grad_output.shape[1] * input.shape[1] + (threads - 1)) // threads
    # Call internal function to calcaluate actually
    gradient_linear_weight_internal_2d[blockspergrid, threads](grad_weight, grad_output, input)

    return grad_weight



# TODO : make another function to just grouping tensor...?
@cuda.jit
def get_precision_internal(v, res, bs, gd):
    
    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x    
    
    i0g =  idx // (bs[3]*bs[2]*bs[1])
    i1g = (idx // (bs[3]*bs[2])) % bs[1]
    i2g = (idx //  bs[3]) % bs[2]
    i3g =  idx % bs[3]

    i0s, i1s, i2s, i3s = i0g*gd[0], i1g*gd[1], i2g*gd[2], i3g*gd[3]
    i0e = i0s+gd[0] if i0s+gd[0] < v.shape[0] else v.shape[0]
    i1e = i1s+gd[1] if i1s+gd[1] < v.shape[1] else v.shape[1]
    i2e = i2s+gd[2] if i2s+gd[2] < v.shape[2] else v.shape[2]
    i3e = i3s+gd[3] if i3s+gd[3] < v.shape[3] else v.shape[3]
    # Find the max exponent from each group
    M = 0
    for idx0 in range(i0s, i0e):
        for idx1 in range(i1s, i1e):
            for idx2 in range(i2s, i2e):
                for idx3 in range(i3s, i3e):
                    res[idx0,idx1,idx2,idx3] = (v[idx0,idx1,idx2,idx3] & 0x7f800000 ) >> 23

    cuda.syncthreads()


# get_zse : Get ZSE of a 4d tensor
def get_precision(inp, group_mantissa, group_dim):
    print("get_precision executed:")
    # Make true to ZSE analyze, temporal disabled
    # Set pointer of tensor as int, easier to manipulate
    inp_ = inp.view(torch.int32)
    ins = np.array(inp.size())
    # Array to store result
    
    if len(ins) == 4: # If the tensor size is 4d
        # Choose Ideal thread size

        # Set blockspergrid and call internal function
        bs = ((ins[0]-1)//group_dim[0]+1, (ins[1]-1)//group_dim[1]+1, (ins[2]-1)//group_dim[2]+1, (ins[3]-1)//group_dim[3]+1)
        inpsize = (ins[0], ins[1], ins[2], ins[3])
        
        res = np.zeros(inp.size())

        threads = np.product(bs)
        threads = threads + (32 - threads % 32)
        if threads > 512: # Adjusted here!
            threads = 512
        threads = int(threads)
        blockspergrid = (np.product(bs) +  (threads - 1)) // threads
        
        # args = (bs[0], bs[1], bs[2], bs[3], group_dim[0], group_dim[1], group_dim[2], group_dim[3], group_mantissa)

        # print(ins, group_mantissa, group_dim, bs, res.shape, threads, blockspergrid, end="  ")
        get_precision_internal[blockspergrid, threads](inp_, res, bs, group_dim)

    else: # Tensor dimension is not supported
        print("get_zse ERROR: Tensor dimension not supported %s"%(str(inp.size())))
    s = "Data:\n"
    for i0 in range(inp.shape[0]):
        for i1 in range(inp.shape[1]):
            s += " ======== %d, %d ========\n"%(i0, i1)
            for i2 in range(inp.shape[2]):
                for i3 in range(inp.shape[3]):
                    s += "%4.2f %3d\t"%(inp[i0,i1,i2,i3], res[i0,i1,i2,i3])
            s += "\n"
    print(s)

    return a


# TODO : make another function to just grouping tensor...?
@cuda.jit
def get_zse_internal(v, res, bs, gd, gm):

    idx = cuda.threadIdx.x + cuda.blockDim.x  * cuda.blockIdx.x    
    
    i0g =  idx // (bs[3]*bs[2]*bs[1])
    i1g = (idx // (bs[3]*bs[2])) % bs[1]
    i2g = (idx //  bs[3]) % bs[2]
    i3g =  idx % bs[3]

    i0s, i1s, i2s, i3s = i0g*gd[0], i1g*gd[1], i2g*gd[2], i3g*gd[3]
    i0e = i0s+gd[0] if i0s+gd[0] < v.shape[0] else v.shape[0]
    i1e = i1s+gd[1] if i1s+gd[1] < v.shape[1] else v.shape[1]
    i2e = i2s+gd[2] if i2s+gd[2] < v.shape[2] else v.shape[2]
    i3e = i3s+gd[3] if i3s+gd[3] < v.shape[3] else v.shape[3]
    # Find the max exponent from each group
    M = 0
    for idx0 in range(i0s, i0e):
        for idx1 in range(i1s, i1e):
            for idx2 in range(i2s, i2e):
                for idx3 in range(i3s, i3e):
                    e = (v[idx0,idx1,idx2,idx3] & 0x7f800000 ) >> 23
                    if M < e:
                        M = e
    if M == 0:
        return

    rs = 0
    # Remove each mantissa to desired values
    for idx0 in range(i0s, i0e):
        for idx1 in range(i1s, i1e):
            for idx2 in range(i2s, i2e):
                for idx3 in range(i3s, i3e):
                    e = (v[idx0,idx1,idx2,idx3] & 0x7f800000 ) >> 23
                    if M - e + 1 > gm:
                        rs += 1
    res[i0g, i1g, i2g, i3g] = rs
    cuda.syncthreads()


# get_zse : Get ZSE of a 4d tensor
def get_zse(inp, group_mantissa, group_dim):

    # Make true to ZSE analyze, temporal disabled
    # Set pointer of tensor as int, easier to manipulate
    inp_ = inp.view(torch.int32)
    ins = np.array(inp.size())
    # Array to store result
    
    if len(ins) == 4: # If the tensor size is 4d
        # Choose Ideal thread size

        # Set blockspergrid and call internal function
        bs = ((ins[0]-1)//group_dim[0]+1, (ins[1]-1)//group_dim[1]+1, (ins[2]-1)//group_dim[2]+1, (ins[3]-1)//group_dim[3]+1)
        inpsize = (ins[0], ins[1], ins[2], ins[3])
        
        res = np.zeros(bs)

        threads = np.product(bs)
        threads = threads + (32 - threads % 32)
        if threads > 512: # Adjusted here!
            threads = 512
        threads = int(threads)
        blockspergrid = (np.product(bs) +  (threads - 1)) // threads
        
        # args = (bs[0], bs[1], bs[2], bs[3], group_dim[0], group_dim[1], group_dim[2], group_dim[3], group_mantissa)

        # print(ins, group_mantissa, group_dim, bs, res.shape, threads, blockspergrid, end="  ")
        get_zse_internal[blockspergrid, threads](inp_, res, bs, group_dim, group_mantissa)

    else: # Tensor dimension is not supported
        print("get_zse ERROR: Tensor dimension not supported %s"%(str(inp.size())))

    # print(np.sum(res) / np.prod(ins))

    a = np.sum(res) / np.prod(ins)
    # print("%14.5f  %14d  %14d"%(a, np.sum(res), np.prod(ins)))
    return a
>>>>>>> f8b9006c5864df8a28775e01baa102859ed5816b
