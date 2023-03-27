SHELL := /bin/bash

# ==========================
# Compiler & common options
# ==========================

COMPILER = chpl
CHPL_COMMONS_DIR = ./commons
CHPL_DATA_STRUCT_DIR = ./DistBag-DFS

CHPL_COMMON_OPTS = --fast -M $(CHPL_COMMONS_DIR) -M $(CHPL_DATA_STRUCT_DIR)

# ==========================
# Build Chapel codes
# ==========================

all: main_pfsp.o main_uts.o main_nqueens.o main_knapsack.o

# ==========
# PFSP
# ==========

CHPL_PFSP_MODULES_DIR = ./benchmarks/PFSP
CHPL_PFSP_OPTS = -M $(CHPL_PFSP_MODULES_DIR) -M $(CHPL_PFSP_MODULES_DIR)/instances -suseNewArrayFind

main_pfsp.o: main_pfsp.chpl
	$(COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_PFSP_OPTS) main_pfsp.chpl -o main_pfsp.o

# ==========
# UTS
# ==========

CHPL_UTS_MODULES_DIR = ./benchmarks/UTS

# Default random number generator (RNG)
ifndef RNG
RNG=BRG
endif

RNG_SRC_DIR  = $(CHPL_UTS_MODULES_DIR)/c_sources
RNG_INCL_DIR = $(CHPL_UTS_MODULES_DIR)/c_headers

ifeq ($(RNG), BRG)
RNG_SRC = $(RNG_SRC_DIR)/brg_sha1.c
RNG_INCL= $(RNG_INCL_DIR)/brg_sha1.h
RNG_DEF = -DBRG_RNG
endif
ifeq ($(RNG), ALFG)
RNG_SRC = $(RNG_SRC_DIR)/alfg.c
RNG_INCL= $(RNG_INCL_DIR)/alfg.h
RNG_DEF = -DUTS_ALFG
endif

C_FILES = $(RNG_SRC) $(RNG_INCL)
C_OPTS = --ccflags $(RNG_DEF)

CHPL_UTS_OPTS = -M $(CHPL_UTS_MODULES_DIR) $(C_OPTS) $(C_FILES)

main_uts.o: main_uts.chpl
	$(COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_UTS_OPTS) main_uts.chpl -o main_uts.o

# ==========
# NQueens
# ==========

CHPL_NQUEENS_MODULES_DIR = ./benchmarks/NQueens
CHPL_NQUEENS_OPTS = -M $(CHPL_NQUEENS_MODULES_DIR)

main_nqueens.o: main_nqueens.chpl
	$(COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_NQUEENS_OPTS) main_nqueens.chpl -o main_nqueens.o

# ==========
# Knapsack
# ==========

CHPL_KNAPSACK_MODULES_DIR = ./benchmarks/Knapsack
CHPL_KNAPSACK_OPTS = -M $(CHPL_KNAPSACK_MODULES_DIR)

main_knapsack.o: main_knapsack.chpl
	$(COMPILER) $(CHPL_COMMON_OPTS) $(CHPL_KNAPSACK_OPTS) main_knapsack.chpl -o main_knapsack.o

# ==========================
# Utilities
# ==========================

.PHONY: clean

clean:
	rm main_*.o
	rm main_*.o_real
