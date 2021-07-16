##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (umpire_strategy_headers
  allocation_strategy.hpp
  slot_pool.hpp
  dynamic_pool_map.hpp
  dynamic_pool_list.hpp
  quick_pool.hpp
  thread_safe.hpp)

if (ENABLE_NUMA)
  set (umpire_strategy_headers
    ${umpire_strategy_headers}
    NumaPolicy.hpp)
endif ()

set (umpire_strategy_mixin_headers
  mixins/Inspector.hpp)

set (umpire_strategy_sources)

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
  HEADERS ${umpire_strategy_headers})
  # SOURCES ${umpire_strategy_sources}
  # DEPENDS_ON ${umpire_strategy_depends}
  # OBJECT TRUE)

target_include_directories(
  umpire_strategy
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(
  umpire_strategy
  INTERFACE
  ${umpire_strategy_depends})

install(TARGETS
  umpire_strategy
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
  
install(FILES
  ${umpire_strategy_headers}
  DESTINATION include/umpire/strategy)

install(FILES
  ${umpire_strategy_mixin_headers}
  DESTINATION include/umpire/strategy/mixins)
