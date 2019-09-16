#!/bin/bash
##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

BUILD_DIRECTORY="build_${SYS_TYPE}_${COMPILER}"
CCONF="host-configs/${SYS_TYPE}/${COMPILER}.cmake" 

rm -rf ${BUILD_DIRECTORY} 2>/dev/null
mkdir ${BUILD_DIRECTORY} && cd ${BUILD_DIRECTORY}

cmake \
  -C ../.gitlab/conf/${CCONF} \
  -C ../${CCONF} \
  ..
cmake --build . -j 4
make test
