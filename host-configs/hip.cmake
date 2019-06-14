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

set(ENABLE_HIP ON CACHE BOOL "")
set(ENABLE_OPENMP OFF CACHE BOOL "")
set(ENABLE_CUDA Off CACHE BOOL "")

set(HIP_HIPCC_FLAGS "--amdgpu-target=gfx900" CACHE STRING "")
set(HIP_ROOT_DIR "/opt/rocm/hip" CACHE PATH "HIP ROOT directory path")
