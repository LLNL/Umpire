##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-coral-2017.09.18/bin/clang++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/clang/clang-coral-2017.09.18/bin/clang" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/tce/packages/xl/xl-beta-2017.09.13/bin/xlf2003_r" CACHE PATH "")
set(BLT_EXE_LINKER_FLAGS "-Wl,-rpath,/usr/tce/packages/xl/xl-beta-2017.09.13/lib" CACHE STRING "")
