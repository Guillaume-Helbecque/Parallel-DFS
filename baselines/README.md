# Baseline implementations

## N-Queens C+CUDA

We implemented two C+CUDA versions of the single-core single-GPU accelerated N-Queens
search: `nqueens_cuda.cu` uses explicit data transfers between host and device, while
`nqueens_unified_mem_cuda.cu` exploits unified memory features.
Nodes are managed using a hand-coded work pool.

To compile and execute:
```
make
./nqueens[_unified_mem]_cuda.o -N value -g value -m value -M value
```
where:
- `N` is the number of queens;
- `g` is the number of safety check(s) per evaluation;
- `m` is the minimum number of elements to offload on GPUs;
- `M` is the maximum number of elements to offload on GPUs.

**Note:** By default, the target architecture for C code generation is set to
`-arch=sm_60` in `makefile`.
