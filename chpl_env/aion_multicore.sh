#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --exclusive

# Configuration of the Chapel's environment for multi-core experiments on the
# Aion cluster of the Université du Luxembourg.

# Load the foss toolchain to get access to gcc, mpi, etc...
module load toolchain/foss/2020b

export CHPL_VERSION="1.31.0"
export CHPL_HOME="${PWD}/chapel-${CHPL_VERSION}"

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR:$CHPL_HOME/util"

export CHPL_HOST_PLATFORM="linux64"
export CHPL_HOST_COMPILER="gnu"
export CHPL_LLVM=none
export CHPL_RT_NUM_THREADS_PER_LOCALE=${SLURM_CPUS_PER_TASK}

export GASNET_PHYSMEM_MAX='64 GB'

# if Chapel's directory not found, download it.
if [ ! -d "$CHPL_HOME" ]; then
    module load devel/CMake
    wget -c https://github.com/chapel-lang/chapel/releases/download/${CHPL_VERSION}/chapel-${CHPL_VERSION}.tar.gz -O - | tar xz
    cd chapel-${CHPL_VERSION}
    make -j ${SLURM_CPUS_PER_TASK}
    cd ..
fi
