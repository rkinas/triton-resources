# Puzzle 1: constant add

This challenge is the first puzzle in our Daily Triton Challenge series. The goal is to write a Triton kernel that adds a constant value to each element of a vector. The key aspects of this puzzle are:

- **One program ID axis:** we use a 1D grid, with a single kernel instance.
- **Block size \(B_0\):** the block size is set equal to the length of the vector \(N_0\), so the kernel processes the entire vector in one go.
- **Verification:** the result is compared against a simple PyTorch implementation.

## Full code example

```python
import time
import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# Triton Kernel for Constant Addition
# ------------------------------------------------------------------------------

@triton.jit
def constant_add_kernel(
    x_ptr,          # Pointer to the input vector x
    constant,       # The constant value to add
    y_ptr,          # Pointer to the output vector y
    N0: tl.constexpr,      # Total number of elements in vector x (and y)
    BLOCK_SIZE: tl.constexpr  # Block size, set equal to N0
):
    # Each kernel instance processes a block of elements.
    # With BLOCK_SIZE equal to N0, only one instance is launched.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N0  # Ensure we don't access out-of-bound indices

    # Load x values, add the constant, and store the result in y
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x + constant
    tl.store(y_ptr + offsets, y, mask=mask)

# ------------------------------------------------------------------------------
# Python Wrapper Function for the Triton Kernel
# ------------------------------------------------------------------------------

def constant_add_triton(x: torch.Tensor, constant: float) -> torch.Tensor:
    """
    Adds a constant to each element of the input vector x using a Triton kernel.
    
    The block size is set equal to the vector length (N0), meaning that only one
    kernel instance is launched.
    
    Args:
        x (torch.Tensor): Input vector on CUDA.
        constant (float): The constant to add to each element.
    
    Returns:
        torch.Tensor: Output vector with the constant added.
    """
    N0 = x.numel()
    BLOCK_SIZE = N0  # Block size equals the vector length
    y = torch.empty_like(x)
    
    # With BLOCK_SIZE = N0, our grid consists of a single block.
    grid = lambda meta: (1,)
    
    # Launch the Triton kernel
    constant_add_kernel[grid](x, constant, y, N0, BLOCK_SIZE=BLOCK_SIZE)
    return y

# ------------------------------------------------------------------------------
# Main: Test the constant add kernel
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Create an example vector on the GPU.
    N0 = 1024  # Length of the vector
    x = torch.arange(0, N0, device='cuda', dtype=torch.float32)
    constant = 3.0  # The constant value to add

    # Compute the result using the Triton kernel.
    y_triton = constant_add_triton(x, constant)

    # Compute the result using PyTorch for verification.
    y_torch = x + constant

    # Verify correctness.
    if torch.allclose(y_triton, y_torch):
        print("Success: Triton kernel result matches PyTorch result!")
    else:
        print("Error: The results do not match.")

    # Benchmark the Triton kernel.
    triton_time = benchmark(constant_add_triton, x, constant)
    print(f"Average execution time (Triton): {triton_time:.3f} ms")
```

## Code explanation

### 1. The Triton kernel (`constant_add_kernel`)
- **Kernel signature:**  
  the kernel receives pointers for the input vector `x`, the constant value to add, and the output vector `y`. It also gets the total number of elements `N0` and a compile-time constant `BLOCK_SIZE`.
  
- **Program ID and offsets:**  
  `pid = tl.program_id(0)` obtains the current program ID along the single grid axis. Using this, the kernel calculates the offsets for each element in the block. Since `BLOCK_SIZE` is set equal to `N0`, only one block (one kernel instance) is needed.
  
- **Boundary mask:**  
  a mask (`mask = offsets < N0`) ensures safe memory accesses.
  
- **Addition operation:**  
  the kernel loads the data from `x`, adds the provided constant, and stores the result into `y`.

### 2. Python wrapper function (`constant_add_triton`)
- **Purpose:**  
  this function allocates the output tensor and configures the grid for launching the Triton kernel.
  
- **Grid configuration:**  
  with `BLOCK_SIZE = N0`, the grid is defined as `(1,)` since the entire vector is processed by a single kernel instance.

### 3. Main routine
- **Setup:**  
  a vector `x` of length 1024 is created on the GPU, and a constant value of 3.0 is chosen.
  
- **Validation:**  
  the Triton kernel’s output is compared to PyTorch's built-in addition to ensure correctness.

## Conclusion

Puzzle 1: Constant Add is the first step in our Daily Triton Challenge. This simple yet effective exercise helps you grasp the basic structure of writing a Triton kernel, setting up the grid, and ensuring correct memory operations. As you progress, you'll build on these fundamentals to explore more advanced topics in GPU kernel programming with Triton.
