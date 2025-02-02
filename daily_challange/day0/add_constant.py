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
# Benchmarking Function
# ------------------------------------------------------------------------------

def benchmark(func, *args, n_warmup=10, n_iters=100):
    """
    Benchmarks a function by running warm-up iterations followed by timed iterations.
    
    Args:
        func (callable): The function to benchmark.
        *args: Arguments to pass to the function.
        n_warmup (int): Number of warm-up iterations.
        n_iters (int): Number of iterations for timing.
    
    Returns:
        float: Average execution time per iteration in milliseconds.
    """
    # Warm-up runs to allow for any initial overhead (e.g., CUDA context setup)
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()  # Ensure warm-up runs are complete

    start = time.perf_counter()
    for _ in range(n_iters):
        func(*args)
    torch.cuda.synchronize()  # Wait for all GPU operations to finish
    end = time.perf_counter()
    
    avg_time_ms = (end - start) / n_iters * 1000
    return avg_time_ms

# ------------------------------------------------------------------------------
# Main: Test and Benchmark the Constant Add Kernel
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
