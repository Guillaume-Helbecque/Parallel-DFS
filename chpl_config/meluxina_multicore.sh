#!/bin/bash -l
#SBATCH --time=00:15:00
#SBATCH --account=pxxxxxx
#SBATCH --partition=cpu
#SBATCH --qos=default
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

# Configuration of Chapel for multicore experiments on the Luxembourg national
# MeluXina cluster (https://docs.lxp.lu/).

module load GCC/11.3.0
module load CMake/3.23.1-GCCcore-11.3.0

export CHPL_VERSION="1.33.0"
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
