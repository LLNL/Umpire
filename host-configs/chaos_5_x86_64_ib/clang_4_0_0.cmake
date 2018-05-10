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
set(CMAKE_CXX_COMPILER "/usr/global/tools/clang/chaos_5_x86_64_ib/clang-4.0.0/bin/clang++" CACHE PATH "")
set(CMAKE_C_COMPILER "/usr/global/tools/clang/chaos_5_x86_64_ib/clang-4.0.0/bin/clang" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/usr/apps/gnu/4.9.3/bin/gfortran" CACHE PATH "")

include(${CMAKE_CURRENT_LIST_DIR}/cudatoolkit_8_0.cmake)
