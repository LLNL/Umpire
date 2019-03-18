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
set (umpire_strategy_headers
  AllocationAdvisor.hpp
  AllocationStrategy.hpp
  AllocationTracker.hpp
  MonotonicAllocationStrategy.hpp
  SlotPool.hpp
  SizeLimiter.hpp
  ThreadSafeAllocator.hpp
  DynamicPool.hpp
  DynamicPoolHeuristic.hpp
  FixedPool.hpp
  FixedPool.inl)

if (ENABLE_NUMA)
  set (umpire_strategy_headers
    ${umpire_strategy_headers}
    NumaPolicy.hpp)
endif ()

set (umpire_strategy_mixin_headers
  mixins/Inspector.hpp)

set (umpire_strategy_sources
  AllocationAdvisor.cpp
  AllocationStrategy.cpp
  AllocationTracker.cpp
  MonotonicAllocationStrategy.cpp
  SlotPool.cpp
  SizeLimiter.cpp
  ThreadSafeAllocator.cpp
  DynamicPool.cpp
  DynamicPoolHeuristic.cpp
  mixins/Inspector.cpp)

if (ENABLE_NUMA)
  set (umpire_strategy_sources
    ${umpire_strategy_sources}
    NumaPolicy.cpp)
endif ()

set(umpire_strategy_depends)

if (ENABLE_CUDA)
  set(umpire_strategy_depends
    ${umpire_strategy_depends}
    cuda_runtime)
endif ()

blt_add_library(
  NAME umpire_strategy
  HEADERS ${umpire_strategy_headers} ${umpire_strategy_mixin_headers}
  SOURCES ${umpire_strategy_sources}
  DEPENDS_ON ${umpire_strategy_depends}
  OBJECT TRUE)

target_include_directories(
  umpire_strategy
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(FILES
  ${umpire_strategy_headers}
  DESTINATION include/umpire/strategy)

install(FILES
  ${umpire_strategy_mixin_headers}
  DESTINATION include/umpire/strategy/mixins)
