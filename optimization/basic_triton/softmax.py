import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_kernel(X_ptr,
                         n_cols,
                         output,
                         BLOCK_SIZE: tl.constexpr):
    # the grid here is also a 1D grid, as the row will always fit
    # (block size is next power of two to the number of elements in each row)

    pid = tl.program_id(axis=0)

    # the row start is n_elements * row_index from the input pointer
    row_start = X_ptr + n_cols * pid

    # since we in this simple case always will have the full row, the offsets is just an arange
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    row = tl.load(row_start + offsets, mask=mask, other=-float("inf"))
    row_max = tl.max(row)
    row_value = tl.exp(row - row_max)

    normalization_factor = tl.sum(row_value)

    tl.store(output + n_cols * pid + offsets, row_value / normalization_factor, mask=mask)

    



def softmax(X: torch.Tensor):

    output = torch.empty_like(X)
    n_rows, n_cols = X.shape  # Get actual dimensions

    # Block size should be based on number of columns (elements per row)
    block_size = triton.next_power_of_2(n_cols)
    n_rows, n_cols = X.shape  # Get actual dimensions

    grid = (n_rows,)
    
    fused_softmax_kernel[grid](X, n_cols, output, BLOCK_SIZE=block_size)

    return output



torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)