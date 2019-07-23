##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set(CMAKE_CXX_COMPILER "/usr/tce/packages/intel/intel-17.0.2/bin/icpc" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/intel/intel-17.0.2/bin/icc" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/tce/packages/intel/intel-17.0.2/bin/ifort" CACHE PATH "")

set(COMMON_FLAGS "-gxx-name=/usr/tce/packages/gcc/gcc-4.9.3/bin/g++ -std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_FLAGS} -O3 -finline-functions -axCORE-AVX2 -diag-disable cpu-dispatch" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${COMMON_FLAGS} -O3 -g -finline-functions -axCORE-AVX2 -diag-disable cpu-dispatch" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "${COMMON_FLAGS} -O0 -g" CACHE STRING "")
