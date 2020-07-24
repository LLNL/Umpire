###################
# Generated host-config - Edit at own risk!
###################
# Copyright (c) 2020, Lawrence Livermore National Security, LLC and
# other Umpire Project Developers. See the top-level LICENSE file for
# details.
#
# SPDX-License-Identifier: (BSD-3-Clause) 
###################

#------------------
# SYS_TYPE: toss_3_x86_64_ib
# Compiler Spec: gcc@7.1.0
# CMake executable path: /usr/tce/packages/cmake/cmake-3.14.5/bin/cmake
#------------------

#------------------
# Compilers
#------------------

set(CMAKE_C_COMPILER "/usr/tce/packages/gcc/gcc-7.1.0/bin/gcc" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-7.1.0/bin/g++" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -finline-functions -axCORE-AVX2 -diag-disable cpu-dispatch" CACHE PATH "")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -finline-functions -axCORE-AVX2 -diag-disable cpu-dispatch" CACHE PATH "")

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE PATH "")

#------------------------------------------------------------------------------
# Cuda
#------------------------------------------------------------------------------

set(ENABLE_CUDA ON CACHE BOOL "")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-10.1.168" CACHE PATH "")

set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")

set(ENABLE_C ON CACHE BOOL "")

set(ENABLE_FORTRAN OFF CACHE BOOL "")

set(ENABLE_NUMA OFF CACHE BOOL "")

set(ENABLE_OPENMP OFF CACHE BOOL "")
