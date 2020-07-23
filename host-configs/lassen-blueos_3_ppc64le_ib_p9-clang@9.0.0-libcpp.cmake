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
# SYS_TYPE: blueos_3_ppc64le_ib_p9
# Compiler Spec: clang@9.0.0
# CMake executable path: /usr/tce/packages/cmake/cmake-3.14.5/bin/cmake
#------------------

#------------------
# Compilers
#------------------

set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-9.0.0/bin/clang" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-9.0.0/bin/clang++" CACHE PATH "")

set(CMAKE_C_FLAGS " -DGTEST_HAS_CXXABI_H_=0" CACHE PATH "")

set(CMAKE_CXX_FLAGS " -stdlib=libc++ -DGTEST_HAS_CXXABI_H_=0" CACHE PATH "")

set(ENABLE_CUDA OFF CACHE BOOL "")

