#!/bin/bash

export BUILD_OPTIONS="-DENABLE_CUDA=Off -DENABLE_FORTRAN=Off ${BUILD_OPTIONS}"

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

${SCRIPTPATH}/build_and_test.sh "$@"
