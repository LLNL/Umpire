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
set(umpire_alloc_headers
  MallocAllocator.hpp)

if (ENABLE_CUDA)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    CudaMallocAllocator.hpp
    CudaMallocManagedAllocator.hpp
    CudaPinnedAllocator.hpp)
endif ()

if (ENABLE_ROCM)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    AmAllocAllocator.hpp)
endif ()

blt_add_library(
  NAME umpire_alloc
  HEADERS ${umpire_alloc_headers})

target_include_directories(
  umpire_alloc
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(FILES
  ${umpire_alloc_headers}
  DESTINATION include/umpire/alloc)
