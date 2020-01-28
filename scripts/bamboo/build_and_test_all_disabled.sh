#!/bin/bash
##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

export BUILD_OPTIONS="-DENABLE_CUDA=Off -DENABLE_FORTRAN=Off ${BUILD_OPTIONS}"

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

${SCRIPTPATH}/build_and_test.sh "$@"
