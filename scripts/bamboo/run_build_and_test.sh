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
    echo "Error: Command Failed"
    exit -1
  fi
}

BUILD_DIR=$1
SOURCE_DIR=$2
COMPILER=$3
BUILD_TYPE=$4

echo "Configuring..."
trycmd "cmake -DENABLE_DEVELOPER_DEFAULTS=On \
    -C ${SOURCE_DIR}/.gitlab/conf/host-configs/${SYS_TYPE}/${COMPILER}.cmake \
    -C ${SOURCE_DIR}/host-configs/${SYS_TYPE}/${COMPILER}.cmake \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} ${BUILD_OPTIONS} ${SOURCE_DIR}"

echo "Building..."
trycmd "make VERBOSE=1 -j"

echo "Testing..."
trycmd "ctest --output-on-failure -T Test"
