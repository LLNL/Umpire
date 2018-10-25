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
blt_add_executable(
  NAME malloc
  SOURCES malloc.cxx
  DEPENDS_ON
    umpire)

blt_add_executable(
  NAME strategy_example
  SOURCES strategy_example.cpp
  DEPENDS_ON
    umpire)

if (ENABLE_FORTRAN)
  blt_add_executable(
    NAME allocator_c
    SOURCES allocator.c
    DEPENDS_ON
      umpire
  )
endif ()

blt_add_executable(
  NAME vector_allocator
  SOURCES vector_allocator.cpp
  DEPENDS_ON
    umpire)

if (ENABLE_STATISTICS)
  blt_add_executable(
    NAME statistics_example
    SOURCES statistics.cpp
    DEPENDS_ON umpire)
endif()

blt_add_executable(
  NAME device_allocator_example
  SOURCES device-allocator.cpp
  DEPENDS_ON umpire cuda
)

set_source_files_properties(
  device-allocator.cpp
  PROPERTIES CUDA_SEPARABLE_COMPILATION On)

set_target_properties(
  device_allocator_example
  PROPERTIES CUDA_SEPARABLE_COMPILATION On)

add_subdirectory(tutorial)
