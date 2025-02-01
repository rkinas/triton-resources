import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# Triton Kernel for Vector Addition
# ------------------------------------------------------------------------------

@triton.jit
def vector_add_kernel(
    A_ptr,          # Pointer to first input vector A
    B_ptr,          # Pointer to second input vector B
    C_ptr,          # Pointer to output vector C
    n_elements: tl.constexpr,  # Number of elements in the vectors
    BLOCK_SIZE: tl.constexpr   # Block size (number of elements per program instance)
):
    # Each program instance (kernel instance) computes a block of elements.
    pid = tl.program_id(0)  # 1D grid: get the program id (i.e. block index)
    # Compute the offsets for the current block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to avoid out-of-bound accesses
    mask = offsets < n_elements

    # Load the corresponding elements from A and B
    a = tl.load(A_ptr + offsets, mask=mask)
    b = tl.load(B_ptr + offsets, mask=mask)
    
    # Perform element-wise addition
    c = a + b
    
    # Store the result into the output pointer C
    tl.store(C_ptr + offsets, c, mask=mask)

# ------------------------------------------------------------------------------
# Python Wrapper Function for the Triton Kernel
# ------------------------------------------------------------------------------

def vector_add_triton(A: torch.Tensor, B: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    """
    Adds two vectors A and B using the Triton kernel.

    Args:
        A (torch.Tensor): First input vector (on CUDA).
        B (torch.Tensor): Second input vector (on CUDA).
        BLOCK_SIZE (int): Number of elements per block for the kernel.

    Returns:
        torch.Tensor: Output vector containing the element-wise sum.
    """
    assert A.numel() == B.numel(), "Input vectors must have the same number of elements."
    n_elements = A.numel()
    # Create an empty tensor for the result (same size and device as A)
    C = torch.empty_like(A)
    
    # Define grid: number of blocks needed to cover all elements
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch the kernel
    vector_add_kernel[grid](A, B, C, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return C

# ------------------------------------------------------------------------------
# Main: Compare Triton Kernel with PyTorch Implementation
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Create two example vectors on the GPU
    n = 1024 * 10  # total number of elements
    A = torch.arange(0, n, device='cuda', dtype=torch.float32)
    B = torch.arange(n, 2 * n, device='cuda', dtype=torch.float32)

    # Add the vectors using the Triton kernel
    C_triton = vector_add_triton(A, B)

    # Add the vectors using PyTorch (for verification)
    C_pytorch = A + B

    # Verify that the results are the same
    if torch.allclose(C_triton, C_pytorch):
        print("Success: The Triton kernel result matches the PyTorch result!")
    else:
        print("Error: The results do not match.")

    # Print part of the result for inspection
    print("Result (first 10 elements):", C_triton[:10])
