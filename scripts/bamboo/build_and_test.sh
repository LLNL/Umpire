#!/bin/bash
##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

function runjob
{
  echo $1
  
  $1 << EOF
    echo "Cleaning out previous build..."
    rm -rf ${BUILD_DIR} 2> /dev/null
    mkdir -p ${BUILD_DIR} 2> /dev/null
    cd ${BUILD_DIR}

    echo "Configuring..."

    cmake -DENABLE_DEVELOPER_DEFAULTS=On \
        -C ${SOURCE_DIR}/.gitlab/conf/host-configs/${SYS_TYPE}/${COMPILER}.cmake \
        -C ${SOURCE_DIR}/host-configs/${SYS_TYPE}/${COMPILER}.cmake \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ${BUILD_OPTIONS} ${SOURCE_DIR}

    if [ $? -ne 0 ]; then
      echo "Error"
      exit -1
    fi

    echo "Building..."
    make VERBOSE=1 -j

    if [ $? -ne 0 ]; then
      echo "Error"
      exit -1
    fi

    echo "Testing..."
    ctest --output-on-failure -T Test
    if [ $? -ne 0 ]; then
      echo "Error"
      exit -1
    fi
EOF
}

export COMPILER=${1:-gcc_4_9_3}
export BUILD_TYPE=${2:-Release}
BUILD_DIR="build-${SYS_TYPE}"
SOURCE_DIR="$( cd "$(dirname "$0")" ; git rev-parse --show-toplevel )"

if [[ $HOSTNAME == *manta* ]] || [[ $HOSTNAME == *ansel* ]]; then
  runjob "lalloc 1 "
  if [ $? -ne 0 ]; then
    echo "Error"
    exit -1
  fi
else
  runjob "srun -ppdebug -t 5 -N 1 "
  if [ $? -ne 0 ]; then
    echo "Error"
    exit -1
  fi
fi
