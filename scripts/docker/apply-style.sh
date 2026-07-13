#!/bin/bash
##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

docker run -v `pwd`:/home/axom/workspace axom/compilers:clang-10 bash -c "cd workspace && mkdir -p docker-build-style && cd docker-build-style && cmake -DENABLE_DEVELOPER_DEFAULTS=On -DENABLE_CLANGQUERY=Off -DENABLE_CLANGTIDY=Off -DENABLE_CPPCHECK=Off -DCMAKE_CXX_COMPILER=clang++ .. && make style"
