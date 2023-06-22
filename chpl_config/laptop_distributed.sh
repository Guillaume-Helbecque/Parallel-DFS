#!/bin/bash

# Configuration of Chapel for distributed experiments on laptop. It simulates
# multiple locales within a single workstation and is useful for debugging but
# not for performance.

export HERE=$(pwd)

export CHPL_VERSION=1.31.0
export CHPL_HOME=~/chapel-${CHPL_VERSION}D

# Download Chapel if not found
if [ ! -d "$CHPL_HOME" ]; then
  cd ~
  wget -c https://github.com/chapel-lang/chapel/releases/download/$CHPL_VERSION/chapel-${CHPL_VERSION}.tar.gz -O - | tar xz
  mv chapel-$CHPL_VERSION $CHPL_HOME
fi

CHPL_BIN_SUBDIR=`"$CHPL_HOME"/util/chplenv/chpl_bin_subdir.py`
export PATH="$PATH":"$CHPL_HOME/bin/$CHPL_BIN_SUBDIR"
export MANPATH="$MANPATH":"$CHPL_HOME"/man

export CHPL_HOST_PLATFORM=`$CHPL_HOME/util/chplenv/chpl_platform.py`
export CHPL_HOST_COMPILER=gnu
export CHPL_TARGET_ARCH=native
export CHPL_LAUNCHER=amudprun
export CHPL_LLVM=none
NUM_T_LOCALE=$(cat /proc/cpuinfo | grep processor | wc -l) # hyperthreading
export CHPL_RT_NUM_THREADS_PER_LOCALE=$NUM_T_LOCALE

export CHPL_COMM=gasnet
export CHPL_COMM_SUBSTRATE=udp
export GASNET_SPAWNFN=L

cd $CHPL_HOME
make -j $NUM_T_LOCALE
cd $HERE/..
