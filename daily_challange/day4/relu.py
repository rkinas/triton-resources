import time
import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# Triton Kernel for ReLU Forward Pass
# ------------------------------------------------------------------------------

@triton.jit
def relu_forward_kernel(
    x_ptr,                # Pointer to input tensor x
    y_ptr,                # Pointer to output tensor y
    N: tl.constexpr,      # Total number of elements in x
    BLOCK_SIZE: tl.constexpr  # Number of elements processed per kernel instance
):
    # Get the current program (block) ID.
    pid = tl.program_id(0)
    # Compute offsets for this block.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask for out-of-bound indices.
    mask = offsets < N
    # Load input values.
    x = tl.load(x_ptr + offsets, mask=mask)
    # Compute ReLU: y = max(0, x)
    y = tl.maximum(x, 0.0)
    # Store the result.
    tl.store(y_ptr + offsets, y, mask=mask)

# ------------------------------------------------------------------------------
# Triton Kernel for ReLU Backward Pass
# ------------------------------------------------------------------------------

@triton.jit
def relu_backward_kernel(
    x_ptr,                # Pointer to saved input tensor x (from forward pass)
    grad_output_ptr,      # Pointer to gradient of the output
    grad_input_ptr,       # Pointer to store computed gradient with respect to x
    N: tl.constexpr,      # Total number of elements in x
    BLOCK_SIZE: tl.constexpr  # Number of elements processed per kernel instance
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # Load input values and gradient of output.
    x = tl.load(x_ptr + offsets, mask=mask)
    grad_out = tl.load(grad_output_ptr + offsets, mask=mask)
    # Compute gradient of ReLU:
    # For each element, if x > 0, gradient is grad_out; otherwise, it is 0.
    grad_in = tl.where(x > 0, grad_out, 0.0)
    tl.store(grad_input_ptr + offsets, grad_in, mask=mask)

# ------------------------------------------------------------------------------
# Custom Autograd Function Using Triton Kernels
# ------------------------------------------------------------------------------

class TritonReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
        """
        Forward pass of the ReLU activation using the Triton kernel.
        Saves the input tensor for use in the backward pass.
        """
        N = x.numel()
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        # Launch the forward kernel.
        relu_forward_kernel[grid](x, y, N, BLOCK_SIZE=BLOCK_SIZE)
        # Save input tensor for the backward pass.
        ctx.save_for_backward(x)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """
        Backward pass computes the gradient of the ReLU activation.
        """
        x, = ctx.saved_tensors
        N = x.numel()
        grad_input = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
        BLOCK_SIZE = ctx.BLOCK_SIZE
        # Launch the backward kernel.
        relu_backward_kernel[grid](x, grad_output, grad_input, N, BLOCK_SIZE=BLOCK_SIZE)
        # Return the gradient for x and None for BLOCK_SIZE (not a tensor).
        return grad_input, None

# Convenience function to call our custom autograd ReLU.
def triton_relu(x: torch.Tensor, BLOCK_SIZE: int = 1024) -> torch.Tensor:
    return TritonReLUFunction.apply(x, BLOCK_SIZE)

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
    # Warm-up iterations.
    for _ in range(n_warmup):
        func(*args)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        func(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / n_iters * 1000

# ------------------------------------------------------------------------------
# Main: Test and Benchmark the Autograd-Compatible ReLU
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Create a random input tensor on the GPU with gradient tracking.
    N = 1024 * 1024  # 1 million elements
    x = torch.randn(N, device='cuda', dtype=torch.float32, requires_grad=True)
    BLOCK_SIZE = 1024

    # Forward pass using our custom Triton ReLU.
    y_triton = triton_relu(x, BLOCK_SIZE)
    # Define a dummy loss (sum of outputs) and perform backward pass.
    loss_triton = y_triton.sum()
    loss_triton.backward()
    
    # For validation, compare against PyTorch's built-in ReLU.
    x_torch = x.detach().clone().requires_grad_()
    y_torch = torch.relu(x_torch)
    loss_torch = y_torch.sum()
    loss_torch.backward()

    # Check if the gradients match.
    if torch.allclose(x.grad, x_torch.grad, atol=1e-4):
        print("Success: Triton autograd ReLU backward matches PyTorch!")
    else:
        print("Error: The gradients do not match.")

    # Benchmark the forward pass.
    triton_time = benchmark(lambda: triton_relu(x, BLOCK_SIZE))
    torch_time = benchmark(lambda: torch.relu(x))
    print(f"Average execution time (Forward Pass):")
    print(f"  Triton ReLU = {triton_time:.3f} ms")
    print(f"  PyTorch ReLU = {torch_time:.3f} ms")
