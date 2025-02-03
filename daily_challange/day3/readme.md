# Puzzle 2: ReLU Activation with Triton

In this challenge, you will implement the ReLU (Rectified Linear Unit) activation function using Triton. ReLU is defined as:

\[
\text{ReLU}(x) = \max(0, x)
\]

For each element in the input vector, the kernel computes the maximum between the element and 0. This example compares the custom Triton implementation to PyTorch’s built-in ReLU, and it also includes a benchmarking function for performance measurement.

## Key points

- **1D grid processing:** the kernel uses a one-dimensional grid of program IDs. Each kernel instance processes a block of elements.
- **Block-based computation:** the vector is processed in blocks with a configurable block size.
- **Element-wise operation:** for each element, the kernel computes `y = max(0, x)`.

## Full code example

```python
import time
import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# Triton Kernel for ReLU Activation
# ------------------------------------------------------------------------------

@triton.jit
def relu_kernel(
    x_ptr,               # Pointer to the input vector x
    y_ptr,               # Pointer to the output vector y
    N: tl.constexpr,     # Total number of elements in the input vector
    BLOCK_SIZE: tl.constexpr  # Block size: number of elements processed per kernel instance
):
    # Each kernel instance processes a block of elements.
    # Get the current program ID along the 1D grid.
    pid = tl.program_id(0)
    
    # Compute the offsets for the block of elements this kernel instance will process.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to ensure we do not access out-of-bound memory.
    mask = offsets < N
    
    # Load elements from the input pointer.
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Compute the ReLU activation: y = max(0, x)
    y = tl.maximum(x, 0.0)
    
    # Store the result back to the output pointer.
    tl.store(y_ptr + offsets, y, mask=mask)

# ------------------------------------------------------------------------------
# Python Wrapper Function for the Triton ReLU Kernel
# ------------------------------------------------------------------------------

def relu_triton(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    """
    Applies the ReLU activation function on the input vector x using a Triton kernel.
    
    Args:
        x (torch.Tensor): Input tensor on CUDA.
        BLOCK_SIZE (int): Number of elements processed per kernel instance.
    
    Returns:
        torch.Tensor: Output tensor after applying ReLU activation.
    """
    N = x.numel()
    # Allocate the output tensor with the same shape and device as the input.
    y = torch.empty_like(x)
    
    # Configure the grid: number of blocks required to cover all N elements.
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    # Launch the Triton kernel.
    relu_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
    return y

# ------------------------------------------------------------------------------
# Benchmarking Function
# ------------------------------------------------------------------------------

def benchmark(func, *args, n_warmup=10, n_iters=100):
    """
    Benchmarks a function by performing warm-up iterations followed by timed iterations.
    
    Args:
        func (callable): The function to benchmark.
        *args: Arguments to pass to the function.
        n_warmup (int): Number of warm-up iterations.
        n_iters (int): Number of iterations for timing.
    
    Returns:
        float: Average execution time per iteration in milliseconds.
    """
    # Warm-up: execute the function several times to mitigate initial overhead.
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()  # Wait for all GPU operations to finish.

    # Timing the execution.
    start = time.perf_counter()
    for _ in range(n_iters):
        func(*args)
    torch.cuda.synchronize()  # Ensure all GPU operations are complete.
    end = time.perf_counter()

    avg_time_ms = (end - start) / n_iters * 1000
    return avg_time_ms

# ------------------------------------------------------------------------------
# Main: Test and Benchmark the Triton ReLU Kernel
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Create an example input vector on the GPU.
    N = 1024 * 1024  # For instance, 1 million elements.
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    
    # Apply ReLU using the Triton kernel.
    y_triton = relu_triton(x)
    
    # Apply ReLU using PyTorch for validation.
    y_torch = torch.relu(x)
    
    # Verify that both outputs are the same.
    if torch.allclose(y_triton, y_torch):
        print("Success: Triton ReLU matches PyTorch ReLU!")
    else:
        print("Error: The Triton ReLU output does not match PyTorch.")

    # Benchmark the Triton kernel.
    triton_time = benchmark(relu_triton, x)
    print(f"Average execution time (Triton ReLU): {triton_time:.3f} ms")

    # Benchmark PyTorch’s built-in ReLU.
    torch_time = benchmark(torch.relu, x)
    print(f"Average execution time (PyTorch ReLU): {torch_time:.3f} ms")
```

## Code explanation

### 1. The Triton kernel (`relu_kernel`)
- **Kernel signature:**  
  The kernel takes pointers for the input (`x_ptr`) and output (`y_ptr`) vectors, along with the total number of elements (`N`) and a compile-time constant `BLOCK_SIZE`.
  
- **Program ID and offsets:**  
  The kernel retrieves its program ID using `tl.program_id(0)` and computes the element offsets within the vector for the current block.
  
- **Masking:**  
  A mask is created (`mask = offsets < N`) to prevent out-of-bound memory accesses when the vector size is not an exact multiple of the block size.
  
- **ReLU computation:**  
  The kernel loads the input elements, computes the maximum between each element and 0 using `tl.maximum(x, 0.0)`, and then stores the result.
  
### 2. Python wrapper function (`relu_triton`)
- **Purpose:**  
  This function sets up the output tensor and computes the grid configuration needed to launch the kernel. It then calls the Triton kernel with the correct arguments.
  
- **Grid configuration:**  
  The grid is computed with `triton.cdiv(N, meta['BLOCK_SIZE'])` ensuring all elements are processed even if the total number of elements isn’t an exact multiple of the block size.

### 3. Benchmarking function (`benchmark`)
- **Warm-up iterations:**  
  Several warm-up iterations help avoid measuring the initial overhead such as CUDA context initialization.
  
- **Timing:**  
  The function measures the average execution time over a set number of iterations. Synchronization (`torch.cuda.synchronize()`) is used before and after the timing loop to ensure accurate measurement.

### 4. Main routine
- **Setup:**  
  A large random input vector is generated on the GPU.
  
- **Validation:**  
  The output from the Triton kernel is compared with PyTorch’s `torch.relu` to ensure the correctness of the implementation.
  
- **Benchmarking:**  
  Both the Triton and PyTorch ReLU functions are benchmarked, and their average execution times are printed.

## Conclusion

This puzzle demonstrates how to implement a ReLU activation function using Triton. By comparing it with PyTorch’s implementation and measuring performance, you gain practical insight into writing and optimizing custom GPU kernels. This is another step forward in your Daily Triton Challenge as you explore GPU programming from basic to more advanced operations.
