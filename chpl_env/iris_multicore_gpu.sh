#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH -p gpu
#SBATCH -G 4

# Configuration of the Chapel's environment for multi-core + GPU accelerated
# experiments on the Iris cluster of the Université du Luxembourg.

# Load the foss toolchain to get access to gcc, mpi, etc...
module load toolchain/foss/2020b
module load system/CUDA/11.1

export CHPL_VERSION="1.31.0"
export CHPL_HOME="${PWD}/chapel-${CHPL_VERSION}"

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR:$CHPL_HOME/util"

export CHPL_HOST_PLATFORM="linux64"
export CHPL_HOST_COMPILER="gnu"
export CHPL_LLVM="bundled" # must be used
# export CHPL_RT_NUM_THREADS_PER_LOCALE=${SLURM_CPUS_PER_TASK}
export CHPL_RT_NUM_THREADS_PER_LOCALE=1 # necessary if using CUDA 10
export CHPL_LOCALE_MODEL="gpu"

export GASNET_PHYSMEM_MAX='64 GB'

# if Chapel's directory not found, download it.
if [ ! -d "$CHPL_HOME" ]; then
    module load devel/CMake/3.20.1-GCCcore-10.2.0
    wget -c https://github.com/chapel-lang/chapel/releases/download/${CHPL_VERSION}/chapel-${CHPL_VERSION}.tar.gz -O - | tar xz
    cd chapel-${CHPL_VERSION}
    make -j ${SLURM_CPUS_PER_TASK}
    cd ..
fi
