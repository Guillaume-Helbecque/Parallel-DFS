SHELL := /bin/bash

# Compilers & common options
CHPL_COMPILER := chpl
C_COMPILER    := gcc
CUDA_COMPILER := nvcc
HIP_COMPILER  := hipcc

#CHPL_DATA_STRUCT_DIR := ../DistBag-DFS
CHPL_COMMON_OPTS := --fast
C_COMMON_OPTS    := -O3
CUDA_COMMON_OPTS := $(C_COMMON_OPTS) -arch=sm_60
HIP_COMMON_OPTS  := $(C_COMMON_OPTS) -offload-arch=gfx906

# Source files
CHPL_SOURCES := nqueens_chpl.chpl nqueens_unified_mem_chpl.chpl #nqueens_bag_chpl.chpl
C_SOURCES    := nqueens_c.c
CUDA_SOURCES := nqueens_cuda.cu nqueens_unified_mem_cuda.cu nqueens_multigpu_cuda.cu

# Object files
CHPL_OBJECTS := $(CHPL_SOURCES:.chpl=.o)
C_OBJECTS    := $(C_SOURCES:.c=.o)
CUDA_OBJECTS := $(CUDA_SOURCES:.cu=.o)
HIP_OBJECTS  := $(CUDA_SOURCES:cuda.cu=hip.o)

# Build codes
all: $(CHPL_OBJECTS) $(C_OBJECTS) $(CUDA_OBJECTS) $(HIP_OBJECTS)

# Pattern rule for CHPL source files
%.o: %.chpl
	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) $< -o $@

# nqueens_chpl_bag.o: nqueens_chpl_bag.chpl
# 	$(CHPL_COMPILER) $(CHPL_COMMON_OPTS) -M $(CHPL_DATA_STRUCT_DIR) nqueens_chpl_bag.chpl -o nqueens_chpl_bag.o

# Pattern rule for C source files
%.o: %.c
	$(C_COMPILER) $(C_COMMON_OPTS) $< -o $@

# Pattern rule for CUDA source files
%.o: %.cu
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) $< -o $@

# Pattern rule for HIP source files
%hip.o: %cuda.cu
	hipify-perl $< > $<.hip
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) $<.hip -o $@

# Utilities
.PHONY: clean

clean:
	rm -f $(CHPL_OBJECTS) $(C_OBJECTS) $(CUDA_OBJECTS) $(HIP_OBJECTS) *.hip
