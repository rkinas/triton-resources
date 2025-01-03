<div align="center"><img src="./assets/triton.png" alt="Triton" width="150" height="150"></div>

# Triton OpenAI
A curated list of resources for learning and exploring Triton, OpenAI's programming language for writing efficient GPU code.

## Official Documentation
- [Official Triton Documentation](https://triton-lang.org/main/index.html)

## Articles
Gain deeper insights into Triton through these detailed articles:
- Understanding the Triton Tutorials [Part 1](https://isamu-website.medium.com/understanding-the-triton-tutorials-part-1-6191b59ba4c) and [Part 2](https://isamu-website.medium.com/understanding-triton-tutorials-part-2-f6839ce50ae7)
- [Softmax in OpenAI Triton](http://blog.nagi.fun/triton-intro-softmax) -> more detailed Fused Softmax Triton example explanation (step-by-step)  
- [Accelerating AI with Triton: A Deep Dive into Writing High-Performance GPU Code](https://medium.com/@nijesh-kanjinghat/accelerating-ai-with-triton-a-deep-dive-into-writing-high-performance-gpu-code-a1e4d66556cc)
- [Accelerating Triton Dequantization Kernels for GPTQ](https://pytorch.org/blog/accelerating-triton/)
- [Triton Tutorial #2](https://medium.com/@sherlockliao01/triton-tutorial-2-5de66cd2170d)
- [Triton: OpenAI’s Innovative Programming Language for Custom Deep-Learning Primitives](https://blog.devgenius.io/triton-openais-innovative-programming-language-for-custom-deep-learning-primitives-485723b0b49)
- [Triton Kernel Compilation Stages](https://pytorch.org/blog/triton-kernel-compilation-stages/)
- Deep Dive into Triton Internals [Part 1](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals/), [Part 2](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-2/) and [Part 3](https://www.kapilsharma.dev/posts/deep-dive-into-triton-internals-3/)
- [Exploring Triton GPU programming for neural networks in Java](https://openjdk.org/projects/babylon/articles/triton)
- [Using User-Defined Triton Kernels with torch.compile](https://pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
- [Mamba: The Hard Way](https://srush.github.io/annotated-mamba/hard.html)
- FP8: [Accelerating 2D Dynamic Block Quantized Float8 GEMMs in Triton](https://pytorch.org/blog/accelerating-gemms-triton/)
- FP8: [Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)
- FP8: [Deep Dive on the Hopper TMA Unit for FP8 GEMMs](https://pytorch.org/blog/hopper-tma-unit/)
- [Technical Review on PyTorch2.0 and Triton](https://www.jokeren.tech/slides/Triton_bsc.pdf)
- [Towards Agile Development of Efficient Deep Learning Operators](https://www.jokeren.tech/slides/triton_intel.pdf)
- [Developing Triton Kernels on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html)

## Research Papers
Explore the academic foundation of Triton:
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)

## Videos
Learn by watching these informative videos:
- [Lecture 14: Practitioners Guide to Triton](https://www.youtube.com/watch?v=DdTsX6DQk24) and [notebook](https://github.com/gpu-mode/lectures/blob/main/lecture_014/A_Practitioners_Guide_to_Triton.ipynb)
- [Lecture 29: Triton Internals](https://www.youtube.com/watch?v=njgow_zaJMw)
- [Intro to Triton: Coding Softmax in PyTorch](https://www.youtube.com/watch?v=gyKBN1rnefI)
- [Triton Vector Addition Kernel, part 1: Making the Shift to Parallel Programming](https://www.youtube.com/watch?v=MEZ7XhzTLEg&t)
- [Tiled Matrix Multiplication in Triton - part 1](https://www.youtube.com/watch?v=OnZEBBJvWLU)
- [Flash Attention derived and coded from first principles with Triton (Python)](https://www.youtube.com/watch?v=zy8ChVd_oTM)

## Triton community meetup
Watch Triton community meetups to be up to date with Triton recent topics.
- [2024-11-09](https://youtu.be/N0eiYLWyNpc?si=n9T-X-0UaK3j1fXQ)

## Triton-Puzzles
Challenge yourself with these engaging puzzles:
- [To Solve](https://github.com/srush/Triton-Puzzles)
- [Solved](https://github.com/alexzhang13/Triton-Puzzles-Solutions/blob/main/Triton_Puzzles_Solutions_alexzhang13.ipynb)

## Tools
Enhance your Triton development workflow with these tools:
- [Triton Profiler](https://github.com/triton-lang/triton/tree/c5a14cc00598014b303eebac831f19e8a66e9e1d/third_party/proton) and video explaining how to use it [Dev Tools: Proton/Interpreter](https://www.youtube.com/watch?v=Av1za_0o2Qs)
- [Triton-Viz: A Visualization Toolkit for Programming with Triton](https://github.com/Deep-Learning-Profiling-Tools/triton-viz)
- [Make Triton easier - Triton-util provides simple higher-level abstractions for frequent but repetitive tasks. This allows you to write code that is closer to how you actually think.](https://github.com/UmerHA/triton_util/tree/main)
- [TritonBench is a collection of PyTorch operators used to evaluation the performance of Triton, and its integration with PyTorch.](https://github.com/pytorch-labs/tritonbench)

## Conferences
Catch up on the latest advancements from Triton Conferences:
- [2024 Conference Playlist](https://www.youtube.com/watch?v=nglpa_6cYYI&list=PLc_vA1r0qoiTjlrINKUuFrI8Ptoopm8Vz)
- [2023 Conference Playlist](https://www.youtube.com/watch?v=ZGU0Yw7mORE&list=PLc_vA1r0qoiRZfUC3o4_yjj0FtWvodKAz)

## Sample Kernels
Explore practical implementations with these sample kernels:
- [attorch is a subset of PyTorch's nn module, written purely in Python using OpenAI's Triton](https://github.com/BobMcDear/attorch)
- [FlagGems is a high-performance general operator library implemented in OpenAI Triton. It aims to provide a suite of kernel functions to accelerate LLM training and inference.](https://github.com/FlagOpen/FlagGems)
- [Kernl lets you run Pytorch transformer models several times faster on GPU with a single line of code, and is designed to be easily hackable.](https://github.com/ELS-RD/kernl)
- [Linger-Kernel](https://github.com/linkedin/Liger-Kernel)
- [Triton Kernels for Efficient Low-Bit Matrix Multiplication](https://github.com/mobiusml/gemlite)
- [Unsloth Kernels](https://github.com/unslothai/unsloth/tree/main/unsloth/kernels)
- [This is attempt at implementing a Triton kernel for GPTQ inference. This code is based on the GPTQ-for-LLaMa codebase, which is itself based on the GPTQ codebase.](https://github.com/fpgaminer/GPTQ-triton)
- [triton-index - Catalog openly available Triton kernels](https://github.com/gpu-mode/triton-index)
- [Triton-based implementation of Sparse Mixture-of-Experts (SMoE) on GPUs](https://github.com/shawntan/scattermoe)
- [Variety of Triton and CUDA kernels for training and inference](https://github.com/pytorch-labs/applied-ai)
- [EquiTriton is a project that seeks to implement high-performance kernels for commonly used building blocks in equivariant neural networks, enabling compute efficient training and inference](https://github.com/IntelLabs/EquiTriton)
- [Expanded collection of Neural Network activation functions and other function kernels in Triton by OpenAI.](https://github.com/dtunai/triton-activations)
- [Fused kernels](https://github.com/kapilsh/cuda-mode-lecture)
- [Triton activations](https://github.com/dtunai/triton-activations/tree/main) only feed forward
- [LightLLM is a Python-based LLM (Large Language Model) inference and serving framework, notable for its lightweight design, easy scalability, and high-speed performance](https://github.com/ModelTC/lightllm/tree/main/lightllm/common/basemodel/triton_kernel)

## Triton integrations 
- [jax-triton](https://github.com/jax-ml/jax-triton)

## Triton backends
- [Intel® XPU Backend for Triton](https://github.com/intel/intel-xpu-backend-for-triton)

## Triton communities
- [CUDA-MODE](https://discord.gg/gpumode)
---
### Contribution
Feel free to contribute more resources or suggest updates by opening a pull request or issue in this repository.

---
### License
This resource list is open-sourced under the MIT license. 
