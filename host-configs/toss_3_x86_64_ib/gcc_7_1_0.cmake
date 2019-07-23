##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-7.1.0/bin/g++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/gcc/gcc-7.1.0/bin/gcc" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-7.1.0/bin/gfortran" CACHE PATH "")

include(${CMAKE_CURRENT_LIST_DIR}/cudatoolkit_9_1.cmake)
