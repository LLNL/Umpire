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
set(CMAKE_CXX_COMPILER "/usr/tce/packages/gcc/gcc-4.9.3/bin/g++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/tce/packages/gcc/gcc-4.9.3/bin/gcc" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/tce/packages/gcc/gcc-4.9.3/bin/gfortran" CACHE PATH "")

include(${CMAKE_CURRENT_LIST_DIR}/cudatoolkit_9_1.cmake)
