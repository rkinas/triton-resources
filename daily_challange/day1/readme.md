# Vector Addition with Triton and PyTorch

This repository contains a simple example of how to add two vectors using a custom GPU kernel written in [Triton](https://github.com/openai/triton) and compares the result to a standard PyTorch implementation. The result of both implementations is the same.

## Overview

- **Triton Kernel:** A small GPU kernel that divides the input vectors into blocks. Each kernel instance computes the addition for a block of elements.
- **PyTorch Implementation:** A simple element‑wise addition using PyTorch’s built-in tensor operations.

This example demonstrates how to write a Triton kernel, launch it from Python, and verify that the computed result is identical to that of PyTorch.

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/) (with CUDA support)
- [Triton](https://github.com/openai/triton)  
  Install via pip:

  ```bash
  pip install triton
  ```

## Code

Below is the full code example:

```python
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
```

## Code Explanation

### 1. The Triton Kernel (`vector_add_kernel`)
- **Kernel Signature:**  
  The kernel receives pointers to the input arrays (`A_ptr` and `B_ptr`), a pointer for the output array (`C_ptr`), the total number of elements (`n_elements`), and a compile-time constant `BLOCK_SIZE`.
  
- **Kernel Indexing:**  
  `pid = tl.program_id(0)` retrieves the unique index for the current block. Using this, we compute the starting offsets for each block.
  
- **Boundary Checking:**  
  A mask (`mask = offsets < n_elements`) is used to ensure that only valid elements are loaded and stored, which is important when the total number of elements is not a multiple of `BLOCK_SIZE`.
  
- **Memory Operations:**  
  The `tl.load` function reads elements from memory, and `tl.store` writes the computed result back.

### 2. Python Wrapper Function (`vector_add_triton`)
- **Input Validation:**  
  We ensure both input vectors have the same number of elements.
  
- **Result Tensor:**  
  An output tensor `C` is allocated with the same shape and device as the input vectors.
  
- **Kernel Launch Configuration:**  
  The grid is computed using `triton.cdiv(n_elements, meta['BLOCK_SIZE'])` which determines how many blocks are needed.
  
- **Kernel Launch:**  
  The Triton kernel is launched with the computed grid and the provided parameters.

### 3. PyTorch Comparison
- **PyTorch Addition:**  
  The same vector addition is performed using PyTorch's built-in operator (`A + B`).
  
- **Verification:**  
  `torch.allclose` checks that the results from both methods are nearly identical.

## Conclusion

This example demonstrates a minimal Triton kernel for vector addition. Triton allows you to write custom GPU kernels in Python with a syntax similar to CUDA, enabling you to optimize performance-critical operations. The comparison with PyTorch’s built-in vector addition shows that custom kernels can be both simple to write and produce correct results.

Feel free to clone this repository, experiment with different block sizes, and extend this example to more complex operations.
