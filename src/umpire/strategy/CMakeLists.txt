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
set (umpire_strategy_headers
  AllocationAdvisor.hpp
  AllocationStrategy.hpp
  MonotonicAllocationStrategy.hpp
  SlotPool.hpp
  ThreadSafeAllocator.hpp
  DynamicPool.hpp
  FixedPool.hpp
  FixedPool.inl)

set (umpire_stategy_sources
  AllocationAdvisor.cpp
  AllocationStrategy.cpp
  MonotonicAllocationStrategy.cpp
  SlotPool.cpp
  ThreadSafeAllocator.cpp
  DynamicPool.cpp)

set (umpire_strategy_depends
  umpire
  umpire_util
  umpire_tpl_simpool)

blt_add_library(
  NAME umpire_strategy
  HEADERS ${umpire_strategy_headers}
  SOURCES ${umpire_stategy_sources}
  HEADERS_OUTPUT_SUBDIR umpire/strategy
  DEPENDS_ON ${umpire_strategy_depends})

target_include_directories(
  umpire_strategy
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS
  umpire_strategy
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES
  ${umpire_strategy_headers}
  DESTINATION include/umpire/strategy)
