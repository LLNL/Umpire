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

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

export UMPIRE_DIR=$(git rev-parse --show-toplevel)
export BUILD_DIR=build-${SYS_TYPE}

export COMPILER=${1:-gcc_4_9_3}
export BUILD_TYPE=${2:-Release}

mkdir ${BUILD_DIR} 2> /dev/null
cd ${BUILD_DIR}

echo "Configuring..."

trycmd "cmake -C ${UMPIRE_DIR}/host-configs/${SYS_TYPE}/${COMPILER}.cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ${BUILD_OPTIONS} ../"

echo "Building..."
trycmd "make VERBOSE=1 -j"

#
# TODO (MJM) - I'm not sure how to obtain exit status for programs run under bsub and srun
#
echo "Testing..."
if [[ $HOSTNAME == *manta* ]]; then
  bsub -x -n 1 -G guests -Ip ctest --output-on-failure -T Test
else
  srun -ppdebug -t 5 -N 1 ctest --output-on-failure -T Test
fi
