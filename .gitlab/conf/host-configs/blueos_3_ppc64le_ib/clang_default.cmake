##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set(CMAKE_CXX_COMPILER "clang++" CACHE PATH "")
set(CMAKE_C_COMPILER "clang" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/tce/packages/xl/xl-2019.08.20/bin/xlf2003" CACHE PATH "")
set(BLT_EXE_LINKER_FLAGS "-Wl,-rpath,/usr/tce/packages/xl/xl-2019.08.20/lib" CACHE STRING "")
