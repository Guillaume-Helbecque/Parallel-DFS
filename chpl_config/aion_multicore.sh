#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --exclusive

# Configuration of Chapel for multi-core experiments on the Aion cluster of the
# Université du Luxembourg (https://hpc-docs.uni.lu/systems/aion/).

# Load the foss toolchain to get access to gcc, mpi, etc...
module load toolchain/foss/2020b
module load devel/CMake

export CHPL_VERSION="1.31.0"
export CHPL_HOME="$PWD/chapel-${CHPL_VERSION}MC"

# Download Chapel if not found
if [ ! -d "$CHPL_HOME" ]; then
    wget -c https://github.com/chapel-lang/chapel/releases/download/$CHPL_VERSION/chapel-${CHPL_VERSION}.tar.gz -O - | tar xz
    mv chapel-$CHPL_VERSION $CHPL_HOME
fi

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR:$CHPL_HOME/util"

export CHPL_HOST_PLATFORM="linux64"
export CHPL_HOST_COMPILER="gnu"
export CHPL_LLVM="none"
export CHPL_RT_NUM_THREADS_PER_LOCALE=$SLURM_CPUS_PER_TASK

export GASNET_PHYSMEM_MAX='64 GB'

cd $CHPL_HOME
make -j $SLURM_CPUS_PER_TASK
cd ../..
