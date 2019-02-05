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

if (ENABLE_C_API)
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

blt_add_executable(
  NAME size_limiter
  SOURCES size_limiter.cpp
  DEPENDS_ON 
    umpire)

if (ENABLE_STATISTICS)
  blt_add_executable(
    NAME statistics_example
    SOURCES statistics.cpp
    DEPENDS_ON umpire)
endif()

blt_add_executable(
  NAME mixed_pool_example
  SOURCES mixed_pool.cpp
  DEPENDS_ON umpire)

add_subdirectory(tutorial)

add_subdirectory(cookbook)
