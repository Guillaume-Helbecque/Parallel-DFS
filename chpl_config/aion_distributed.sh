#!/bin/bash -l
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:00:00
#SBATCH --exclusive

# Configuration of Chapel for distributed experiments on the Aion cluster of the
# Université du Luxembourg (https://hpc-docs.uni.lu/systems/aion/). Computer nodes
# are interconnected through a Fast InfiniBand (IB) HDR100 network.

# Load the foss toolchain to get access to gcc, mpi, etc...
module load toolchain/foss/2020b
module load devel/CMake

export CHPL_VERSION="1.33.0"
export CHPL_HOME="$PWD/chapel-${CHPL_VERSION}D"

# Download Chapel if not found
if [ ! -d "$CHPL_HOME" ]; then
    wget -c https://github.com/chapel-lang/chapel/releases/download/$CHPL_VERSION/chapel-${CHPL_VERSION}.tar.gz -O - | tar xz
    mv chapel-$CHPL_VERSION $CHPL_HOME
fi

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR:$CHPL_HOME/util"

export CHPL_HOST_PLATFORM="linux64"
export CHPL_HOST_COMPILER="gnu"
export CHPL_LAUNCHER="gasnetrun_ibv"
export CHPL_LLVM="none"
export CHPL_RT_NUM_THREADS_PER_LOCALE=$SLURM_CPUS_PER_TASK

export CHPL_COMM='gasnet'
export CHPL_COMM_SUBSTRATE='ibv'
export CHPL_TARGET_CPU='native'
export GASNET_QUIET="1"
export HFI_NO_CPUAFFINITY="1"

export GASNET_IBV_SPAWNER="ssh"
export GASNET_SSH_SERVERS=`scontrol show hostnames | xargs echo`

export GASNET_PHYSMEM_MAX='64 GB'

cd $CHPL_HOME
make -j $SLURM_CPUS_PER_TASK
cd ../..
