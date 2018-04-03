##############################################################################
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
set(CMAKE_CXX_COMPILER "/usr/local/bin/icpc-16.0.258" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/local/bin/icc-16.0.258" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/local/bin/ifort-16.0.258" CACHE PATH "")

set(COMMON_FLAGS "-gnu-prefix=/usr/apps/gnu/4.9.3/bin/ -Wl,-rpath,/usr/apps/gnu/4.9.3/lib64 -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -O3 -march=native -ansi-alias" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_FLAGS} -O3 -g -march=native -ansi-alias" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS} -O0 -g" CACHE STRING "")

include(${CMAKE_CURRENT_LIST_DIR}/cudatoolkit_8_0.cmake)
