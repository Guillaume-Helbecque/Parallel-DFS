#!/bin/bash

# Configuration of the Chapel's environment for multi-core experiments on laptop.

export HERE=$(pwd)

export CHPL_VERSION=1.31.0
export CHPL_HOME=~/chapel-${CHPL_VERSION}

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR"
export MANPATH="$MANPATH":"$CHPL_HOME"/man

export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`
export CHPL_HOST_COMPILER=gnu
export CHPL_TARGET_ARCH=native
export CHPL_LLVM=none
NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l) # hyperthreading
export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE

# if Chapel's directory not found, download and unpack it.
if [ ! -d "$CHPL_HOME" ]; then
  cd ~
  wget -c https://github.com/chapel-lang/chapel/releases/download/${CHPL_VERSION}/chapel-${CHPL_VERSION}.tar.gz -O - | tar xz
  cd $CHPL_HOME
  make -j ${NUM_T_LOCALE}
  cd $HERE
fi

cd $CHPL_HOME
make -j ${NUM_T_LOCALE}
cd $HERE
