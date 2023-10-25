# Baseline implementations

## N-Queens C+CUDA

We implemented two C+CUDA versions of the single-core single-GPU accelerated N-Queens
search: `nqueens_cuda.cu` uses explicit data transfers between host and device, while
`nqueens_cuda_unified_mem.cu` exploits unified memory features.
Nodes are managed using a hand-coded work pool.

To compile and execute:
```
nvcc -O3 nqueens_cuda[_unified_mem].cu -arch=sm_XX
./a.out -N value -g value -m value -M value
```
where:
- `N` is the number of queens;
- `g` is the number of safety check(s) per evaluation;
- `m` is the minimum number of elements to offload on GPUs;
- `M` is the maximum number of elements to offload on GPUs.
