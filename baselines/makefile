SHELL := /bin/bash

# Compilers & common options
CHPL_COMPILER := chpl
CUDA_COMPILER := nvcc

#CHPL_DATA_STRUCT_DIR := ../DistBag-DFS
CHPL_COMMON_OPTS := --fast
C_COMMON_OPTS := -O3 -arch=sm_60

# Source files
CHPL_SOURCES := nqueens_chpl.chpl nqueens_chpl_unified_mem.chpl #nqueens_chpl_bag.chpl
CUDA_SOURCES := nqueens_cuda.cu nqueens_cuda_unified_mem.cu

# Object files
CHPL_OBJECTS := $(CHPL_SOURCES:.chpl=.o)
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)

# Build codes
all: $(CHPL_OBJECTS) $(CUDA_OBJECTS)

# Pattern rule for CHPL source files
%.o: %.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $< -o $@

# nqueens_chpl_bag.o: nqueens_chpl_bag.chpl
# 	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) -M $(CHPL_DATA_STRUCT_DIR) nqueens_chpl_bag.chpl -o nqueens_chpl_bag.o

# Pattern rule for CUDA source files
%.o: %.cu
	$(CUDA_COMPILER) $(C_COMMON_OPTS) $< -o $@

# Utilities
.PHONY: clean

clean:
	rm -f $(CHPL_OBJECTS) $(CUDA_OBJECTS)
