#!/bin/bash
##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

echo "Cleaning out previous build..."
BUILD_DIR="build-${SYS_TYPE}"
rm -rf ${BUILD_DIR} 2> /dev/null
mkdir -p ${BUILD_DIR} 2> /dev/null
cd ${BUILD_DIR}

export UMPIRE_COMPILER=${1:-gcc_4_9_3}
export UMPIRE_BUILD_TYPE=${2:-Release}
export UMPIRE_SOURCE_DIR="$( cd "$(dirname "$0")" ; git rev-parse --show-toplevel )"

UMPIRE_SCRIPT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"

if [[ $HOSTNAME == *manta* ]] || [[ $HOSTNAME == *ansel* ]]; then
  echo "lalloc 1 ${UMPIRE_SCRIPT_PATH}/run_build_and_test.sh"
  lalloc 1 ${UMPIRE_SCRIPT_PATH}/run_build_and_test.sh
  if [ $? -ne 0 ]; then
    echo "Error: lalloc job failed"
    exit -1
  fi
else
  echo "srun -ppdebug -t 5 -N 1 ${UMPIRE_SCRIPT_PATH}/run_build_and_test.sh"
  srun -ppdebug -t 5 -N 1 ${UMPIRE_SCRIPT_PATH}/run_build_and_test.sh
  if [ $? -ne 0 ]; then
    echo "Error: srun job failed"
    exit -1
  fi
fi
