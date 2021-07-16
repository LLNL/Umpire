##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (umpire_detail_headers
  allocation_map.hpp
  backtrace.hpp
  detect_vendor.hpp
  fixed_malloc_pool.hpp
  io.hpp
  logger.hpp
  memory_map.hpp
  memory_sanitizers.hpp
  mpi.hpp
  output_buffer.hpp
  registry.hpp
  replay.hpp
  )

set (umpire_detail_sources
  allocation_map.cpp
  backtrace.cpp
  detect_vendor.cpp
  fixed_malloc_pool.cpp
  io.cpp
  logger.cpp
  mpi.cpp
  output_buffer.cpp
  replay.cpp
  )

blt_add_library(
  NAME umpire_detail
  HEADERS ${umpire_detail_headers}
  SOURCES ${umpire_detail_sources}
  DEPENDS_ON umpire_tpl_judy camp
  OBJECT TRUE)

target_include_directories(
  umpire_detail
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS
  umpire_detail
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)