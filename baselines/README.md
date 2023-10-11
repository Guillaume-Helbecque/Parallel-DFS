# Baseline implementations

## N-Queens C+CUDA

We implemented a C+CUDA version of the single-core single-GPU accelerated N-Queens search.
Nodes are managed using a hand-coded single pool.

To compile and execute:
```
nvcc -O3 nqueens_cuda.cu -arch=sm_XX
./a.out <N> <g> <minSize> <maxSize>
```
where:
- `N` is the number of queens;
- `g` is the number of safety check(s) per evaluation;
- `minSize` is the minimum number of elements to offload on GPUs;
- `maxSize` is the maximum number of elements to offload on GPUs.
