##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

set(ENABLE_HIP ON CACHE BOOL "")
set(ENABLE_OPENMP OFF CACHE BOOL "")
set(ENABLE_CUDA Off CACHE BOOL "")

set(HIP_HIPCC_FLAGS "--amdgpu-target=gfx900" CACHE STRING "")
set(HIP_ROOT_DIR "/opt/rocm/hip" CACHE PATH "HIP ROOT directory path")
