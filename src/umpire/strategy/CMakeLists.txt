##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (umpire_strategy_headers
  AllocationAdvisor.hpp
  AllocationStrategy.hpp
  AllocationTracker.hpp
  DynamicPool.hpp
  DynamicPoolHeuristic.hpp
  DynamicPoolList.hpp
  DynamicPoolMap.hpp
  DynamicSizePool.hpp
  FixedPool.hpp
  FixedSizePool.hpp
  MixedPool.hpp
  NamedAllocationStrategy.hpp
  MonotonicAllocationStrategy.hpp
  Pool.hpp
  SlotPool.hpp
  SizeLimiter.hpp
  StdAllocator.hpp
  ThreadSafeAllocator.hpp
  ZeroByteHandler.hpp)

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
  DynamicPoolList.cpp
  DynamicPoolMap.cpp
  DynamicPoolHeuristic.cpp
  FixedPool.cpp
  MixedPool.cpp
  mixins/Inspector.cpp
  MonotonicAllocationStrategy.cpp
  NamedAllocationStrategy.cpp
  Pool.cpp
  SlotPool.cpp
  SizeLimiter.cpp
  ThreadSafeAllocator.cpp
  ZeroByteHandler.cpp)

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
