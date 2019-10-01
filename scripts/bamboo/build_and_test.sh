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

function runjob
{
  echo $1
  
  $1 << EOF
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
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ${BUILD_OPTIONS} ${SOURCE_DIR}"

    echo "Building..."
    trycmd "make VERBOSE=1 -j"

    echo "Testing..."
    trycmd "ctest --output-on-failure -T Test"
EOF
}

if [[ $HOSTNAME == *manta* ]] || [[ $HOSTNAME == *ansel* ]]; then
  runjob "lalloc 1 "
else
  runjob "srun -ppdebug -t 5 -N 1 "
fi

