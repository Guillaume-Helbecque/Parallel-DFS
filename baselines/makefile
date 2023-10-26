SHELL := /bin/bash

# ===========================
# Compilers & common options
# ===========================

CUDA_COMPILER = nvcc
CHPL_COMPILER = chpl

CHPL_DATA_STRUCT_DIR = ../DistBag-DFS
C_COMMON_OPTS = -O3 -arch=sm_60
CHPL_COMMON_OPTS = --fast

# ==========================
# Build codes
# ==========================

all: nqueens_chpl.o nqueens_chpl_unified_mem.o nqueens_chpl_bag.o nqueens_cuda.o nqueens_cuda_unified_mem.o

nqueens_chpl.o: nqueens_chpl.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) nqueens_chpl.chpl -o nqueens_chpl.o

nqueens_chpl_unified_mem.o: nqueens_chpl_unified_mem.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) nqueens_chpl_unified_mem.chpl -o nqueens_chpl_unified_mem.o

nqueens_chpl_bag.o: nqueens_chpl_bag.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) -M $(CHPL_DATA_STRUCT_DIR) nqueens_chpl_bag.chpl -o nqueens_chpl_bag.o

nqueens_cuda.o: nqueens_cuda.cu
	$(CUDA_COMPILER) $(C_COMMON_OPTS) nqueens_cuda.cu -o nqueens_cuda.o

nqueens_cuda_unified_mem.o: nqueens_cuda_unified_mem.cu
	$(CUDA_COMPILER) $(C_COMMON_OPTS) nqueens_cuda_unified_mem.cu -o nqueens_cuda_unified_mem.o

# ==========================
# Utilities
# ==========================

.PHONY: clean

clean:
	rm nqueens_*.o
