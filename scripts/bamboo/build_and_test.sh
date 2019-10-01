#!/bin/bash
##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

function trycmd
{
  echo $1
  $1

  if [ $? -ne 0 ]; then
    echo "Error"
    exit -1
  fi
}

BUILD_DIR="build-${SYS_TYPE}"
SOURCE_DIR="$( cd "$(dirname "$0")" ; git rev-parse --show-toplevel )"

echo "Cleaning out previous build..."

rm -rf ${BUILD_DIR} 2> /dev/null
mkdir -p ${BUILD_DIR} 2> /dev/null
cd ${BUILD_DIR}

export COMPILER=${1:-gcc_4_9_3}
export BUILD_TYPE=${2:-Release}

echo "Configuring..."

trycmd "cmake -DENABLE_DEVELOPER_DEFAULTS=On \
	  -C ${SOURCE_DIR}/.gitlab/conf/host-configs/${SYS_TYPE}/${COMPILER}.cmake \
	  -C ${SOURCE_DIR}/host-configs/${SYS_TYPE}/${COMPILER}.cmake \
	  -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ${BUILD_OPTIONS} ../src"

echo "Building..."
trycmd "make VERBOSE=1 -j"

#
# TODO (MJM) - I'm not sure how to obtain exit status for programs run under bsub and srun
#
echo "Testing..."
if [[ $HOSTNAME == *manta* ]] || [[ $HOSTNAME == *ansel* ]]; then
  lalloc 1 ctest --output-on-failure -T Test
else
  srun -ppdebug -t 5 -N 1 ctest --output-on-failure -T Test
fi
