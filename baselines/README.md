# Multi-GPU tree-search to solve N-Queens problem instances

This repository contains Chapel and C+Cuda implementations, including both explicit
data transfers and unified memory variants. Nodes are managed using a hand-coded work pool. The C+Cuda implementations are portable on AMD GPUs using the `hipify-perl` tool.

## Compilation
The provided `makefile` produces all the executables in the following form:
```
nqueens[_unified_mem]_[chpl/cuda/hip].o
```

**Note:** By default, the target architecture for Cuda code generation is set to
`-arch=sm_60`, and to `-offload-arch=gfx906` for AMD. Please adjust these settings
in `makefile` if needed.

## Command-line options
The implementations support the following options:
- `N` is the number of queens;
- `g` is the number of safety check(s) per evaluation;
- `m` is the minimum number of elements to offload on GPU devices;
- `M` is the maximum number of elements to offload on GPU devices;
- `D` is the number of GPU devices.

All these values must be positive integers.
