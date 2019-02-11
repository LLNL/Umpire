#!/bin/bash
##############################################################################
# Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
#
# Created by David Beckingsale, david@llnl.gov
# LLNL-CODE-747640
#
# All rights reserved.
#
# This file is part of Umpire.
#
# For details, see https://github.com/LLNL/Umpire
# Please also see the LICENSE file for MIT license.
##############################################################################

export BUILD_OPTIONS="-DENABLE_CUDA=Off -DENABLE_FORTRAN=Off -DENABLE_IPC=Off -DENABLE_IPC_MPI3=Off ${BUILD_OPTIONS}"

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

${SCRIPTPATH}/build_and_test.sh "$@"
