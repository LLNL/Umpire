#!/bin/bash

UMPIRE_DIR=$(git rev-parse --show-toplevel)
BUILD_DIR=build-${SYS_TYPE}

COMPILER=${1:-gcc_4_9_3}

mkdir ${BUILD_DIR} 2> /dev/null
cd ${BUILD_DIR}

echo "Configuring..."

cmake -C ${UMPIRE_DIR}/host-configs/${SYS_TYPE}/${COMPILER}.cmake ../

echo "Building..."
make -j

echo "Testing..."
${UMPIRE_DIR}/scripts/bamboo/test.sh
