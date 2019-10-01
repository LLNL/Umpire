#!/bin/bash
##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

COMPILER=${1:-gcc_4_9_3}
BUILD_TYPE=${2:-Release}
BUILD_DIR="build-${SYS_TYPE}"
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
SOURCE_DIR="$( cd "$(dirname "$0")" ; git rev-parse --show-toplevel )"

echo "Cleaning out previous build..."
rm -rf ${BUILD_DIR} 2> /dev/null
mkdir -p ${BUILD_DIR} 2> /dev/null
cd ${BUILD_DIR}

if [[ $HOSTNAME == *manta* ]] || [[ $HOSTNAME == *ansel* ]]; then
  echo "lalloc 1 ${SCRIPTPATH}/run_build_and_test.sh ${BUILD_DIR} ${SOURCE_DIR} ${COMPILER} ${BUILD_TYPE}"
  lalloc 1 ${SCRIPTPATH}/run_build_and_test.sh ${BUILD_DIR} ${SOURCE_DIR} ${COMPILER} ${BUILD_TYPE}
  if [ $? -ne 0 ]; then
    echo "Error: lalloc job failed"
    exit -1
  fi
else
  echo "srun -ppdebug -t 5 -N 1 ${SCRIPTPATH}/run_build_and_test.sh ${BUILD_DIR} ${SOURCE_DIR} ${COMPILER} ${BUILD_TYPE}"
  srun -ppdebug -t 5 -N 1 ${SCRIPTPATH}/run_build_and_test.sh ${BUILD_DIR} ${SOURCE_DIR} ${COMPILER} ${BUILD_TYPE}
  if [ $? -ne 0 ]; then
    echo "Error: srun job failed"
    exit -1
  fi
fi
