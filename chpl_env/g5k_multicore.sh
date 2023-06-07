#!/usr/bin/env bash

# Configuration of Chapel for multi-core experiments on the French national
# Grid5000 testbed.

# Load gcc, cuda and cmake
module load gcc/12.2.0_gcc-10.4.0
module load cmake/3.23.3_gcc-10.4.0

export HERE=$(pwd)

export CHPL_VERSION=1.30.0
export CHPL_HOME=~/chapel-${CHPL_VERSION}

# Download the latest Chapel's release if not yet done
if [ ! -d "$CHPL_HOME" ]; then
    cd ~
    wget -c https://github.com/chapel-lang/chapel/releases/download/${CHPL_VERSION}/chapel-${CHPL_VERSION}.tar.gz -O - | tar xz
fi

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR"

export MANPATH="$MANPATH":"$CHPL_HOME"/man

export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`
export CHPL_HOST_COMPILER=gnu
export CHPL_LLVM=none

NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l)
export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE

export GASNET_PHYSMEM_MAX='64 GB'

cd $CHPL_HOME
make -j $NUM_T_LOCALE
cd $HERE/..
