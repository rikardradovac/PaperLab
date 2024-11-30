import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    
    
    # the axis here is 0 because the grid in this simple case is 1D
    pid = tl.program_id(axis=0)
    
    # since we have pointers to the vectors and a fixed block size, we need to create corresponding offsets
    # to point to correct elements in the memory
    
    offsets = tl.arange(0, BLOCK_SIZE) + BLOCK_SIZE * pid
    
    # in order to not try to index out of bound memory, we need a mask
    
    mask = offsets < n_elements
    
    
    # now we can load the vectors into SRAM and compute the addition for this block
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    output = x + y
    
    tl.store(out_ptr + offsets, output, mask=mask)
    

def add(x: torch.Tensor,
        y: torch.Tensor
):
    
    # initialize an empty tensor in shape of input tensor x
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # define the grid of size ceil(n_elements / block_size)
    grid = lambda args: (triton.cdiv(n_elements, args["BLOCK_SIZE"]), )
    
    
    # now we can call the kernel and the input tensors will be converted into pointers
    
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output


torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
    
    
    