#!/bin/bash

export BUILD_OPTIONS="-DENABLE_CUDA=On -DENABLE_FORTRAN=On ${BUILD_OPTIONS}"

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

${SCRIPTPATH}/build_and_test.sh "$@"
