##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (umpire_strategy_headers
  AlignedAllocator.hpp
  AllocationAdvisor.hpp
  AllocationPrefetcher.hpp
  AllocationStrategy.hpp
  DynamicPoolList.hpp
  DynamicSizePool.hpp
  FixedPool.hpp
  FixedSizePool.hpp
  MixedPool.hpp
  MonotonicAllocationStrategy.hpp
  NamedAllocationStrategy.hpp
  NamingShim.hpp
  PoolCoalesceHeuristic.hpp
  QuickPool.hpp
  ResourceAwarePool.hpp
  SizeLimiter.hpp
  SlotPool.hpp
  StdAllocator.hpp
  ThreadSafeAllocator.hpp)

if (UMPIRE_ENABLE_MPI AND UMPIRE_ENABLE_IPC_SHARED_MEMORY AND (UMPIRE_ENABLE_CUDA OR UMPIRE_ENABLE_HIP))
  set (umpire_strategy_headers
    ${umpire_strategy_headers}
    DeviceIpcAllocator.hpp)
endif()

if (UMPIRE_ENABLE_NUMA)
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
  FixedPool.cpp
  MixedPool.cpp
  mixins/AlignedAllocation.cpp
  mixins/AllocateNull.cpp
  mixins/Inspector.cpp
  MonotonicAllocationStrategy.cpp
  NamedAllocationStrategy.cpp
  NamingShim.cpp
  QuickPool.cpp
  ResourceAwarePool.cpp
  SizeLimiter.cpp
  SlotPool.cpp
  ThreadSafeAllocator.cpp)

if (UMPIRE_ENABLE_MPI AND UMPIRE_ENABLE_IPC_SHARED_MEMORY AND (UMPIRE_ENABLE_CUDA OR UMPIRE_ENABLE_HIP))
  set (umpire_strategy_sources
    ${umpire_strategy_sources}
    DeviceIpcAllocator.cpp)
endif()

if (UMPIRE_ENABLE_NUMA)
  set (umpire_strategy_sources
    ${umpire_strategy_sources}
    NumaPolicy.cpp)
endif ()

set(umpire_strategy_depends camp umpire_util)

if (UMPIRE_ENABLE_CUDA)
  set(umpire_strategy_depends
    ${umpire_strategy_depends}
    cuda_runtime)
endif ()

if (UMPIRE_ENABLE_HIP)
  set(umpire_strategy_depends
    ${umpire_strategy_depends}
    blt::hip_runtime)
endif ()

if (UMPIRE_ENABLE_OPENMP_TARGET)
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
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(FILES
  ${umpire_strategy_headers}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/umpire/strategy)

install(FILES
  ${umpire_strategy_mixin_headers}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/umpire/strategy/mixins)

umpire_add_code_checks(PREFIX strategy)
