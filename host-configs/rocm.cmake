##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
#cmake \
#  -DCMAKE_BUILD_TYPE=Release \
#  -DENABLE_HCC=ON -DBLT_SOURCE_DIR=${BLT_DIR} \
#  -DHCC_ARCH=gfx900 \
#  -C ${BLT_DIR}/cmake/blt-test/host-configs/rocm.cmake \
#  -DCMAKE_INSTALL_PREFIX=../install-rocm-release \


set(ENABLE_HCC ON CACHE BOOL "")
set(ENABLE_OPENMP OFF CACHE BOOL "")
set(ENABLE_CUDA Off CACHE BOOL "")
set(ENABLE_GMOCK Off CACHE BOOL "")
set(ENABLE_BENCHMARKS Off CACHE BOOL "")
set(ENABLE_WARNINGS_AS_ERRORS Off CACHE BOOL "")
set(ENABLE_TOOLS Off CACHE BOOL "")

set(HCC_ROOT_DIR "/opt/rocm/" CACHE PATH "ROCm ROOT directory path")

set(HCC_INCLUDE_PATH "${HCC_ROOT_DIR}/hcc/include"  CACHE PATH "")
set(HCC_CXX_LIBRARIES "-L${HCC_ROOT_DIR}/hcc/lib -lhc_am" CACHE STRING "")

set(HCC_ARCH "gfx900" CACHE STRING "")

###########################################################
# specify the target architecture
#  Default with ROCm 1.7 is gfx803 (Fiji)
#  Other options:
#    gfx700  Hawaii
#    gfx803  Polaris (RX580)
#    gfx900  Vega
#    gfx901
###########################################################
set(HCC_ARCH_FLAG "-amdgpu-target=${HCC_ARCH}" CACHE STRING "")

###########################################################
# get compile/link flags from hcc-config
###########################################################
execute_process(COMMAND ${HCC_ROOT_DIR}/hcc/bin/hcc-config --cxxflags OUTPUT_VARIABLE HCC_CXX_COMPILE_FLAGS)
execute_process(COMMAND ${HCC_ROOT_DIR}/hcc/bin/hcc-config --ldflags OUTPUT_VARIABLE HCC_CXX_LINK_FLAGS)

#set(HCC_CXX_COMPILE_FLAGS "${HCC_CXX_COMPILE_FLAGS} -Wno-unused-command-line-argument -DHCC_ENABLE_ACCELERATOR_PRINTF" CACHE STRING "")
set(HCC_CXX_LINK_FLAGS "${HCC_CXX_LINK_FLAGS} ${HCC_ARCH_FLAG} ${HCC_CXX_LIBRARIES}" CACHE STRING "")

###########################################################
# set CMake cache variables
###########################################################
set(CMAKE_CXX_COMPILER "${HCC_ROOT_DIR}/bin/hcc" CACHE FILEPATH "ROCm HCC compiler")
set(BLT_CXX_FLAGS "${HCC_CXX_COMPILE_FLAGS}" CACHE STRING "HCC compiler flags")
set(BLT_EXE_LINKER_FLAGS ${HCC_CXX_LINK_FLAGS} CACHE STRING "")

#set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_COMPILER} ${HCC_CXX_LINK_FLAGS} <OBJECTS> <LINK_LIBRARIES> -o <TARGET>" CACHE STRING "HCC linker command line")
