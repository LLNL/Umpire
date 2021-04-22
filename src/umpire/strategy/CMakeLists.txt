##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (umpire_strategy_headers
  AlignedAllocator.hpp
  AllocationAdvisor.hpp
  AllocationPrefetcher.hpp
  AllocationStrategy.hpp
  DynamicPool.hpp
  DynamicPoolList.hpp
  DynamicPoolMap.hpp
  DynamicSizePool.hpp
  FixedPool.hpp
  FixedSizePool.hpp
  MixedPool.hpp
  MonotonicAllocationStrategy.hpp
  NamedAllocationStrategy.hpp
  QuickPool.hpp
  SizeLimiter.hpp
  SlotPool.hpp
  StdAllocator.hpp
  ThreadSafeAllocator.hpp)

if (ENABLE_NUMA)
  set (umpire_strategy_headers
    ${umpire_strategy_headers}
    NumaPolicy.hpp)
endif ()

set (umpire_strategy_mixin_headers
  mixins/AlignedAllocation.hpp
  mixins/AlignedAllocation.inl
  mixins/AllocateNull.hpp
  mixins/Inspector.hpp)

set (umpire_strategy_sources
  AlignedAllocator.cpp
  AllocationAdvisor.cpp
  AllocationPrefetcher.cpp
  AllocationStrategy.cpp
  DynamicPoolList.cpp
  DynamicPoolMap.cpp
  FixedPool.cpp
  MixedPool.cpp
  mixins/AlignedAllocation.cpp
  mixins/AllocateNull.cpp
  mixins/Inspector.cpp
  MonotonicAllocationStrategy.cpp
  NamedAllocationStrategy.cpp
  QuickPool.cpp
  SizeLimiter.cpp
  SlotPool.cpp
  ThreadSafeAllocator.cpp)

if (ENABLE_NUMA)
  set (umpire_strategy_sources
    ${umpire_strategy_sources}
    NumaPolicy.cpp)
endif ()

set(umpire_strategy_depends camp)

if (ENABLE_CUDA)
  set(umpire_strategy_depends
    ${umpire_strategy_depends}
    cuda_runtime)
endif ()

if (ENABLE_OPENMP_TARGET)
  set( umpire_strategy_depends
    ${umpire_strategy_depends}
    openmp)
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

umpire_add_code_checks(PREFIX strategy)
