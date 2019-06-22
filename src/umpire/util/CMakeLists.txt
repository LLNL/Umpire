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
set (umpire_util_headers
  AllocationMap.hpp
  AllocationRecord.hpp
  Exception.hpp
  FixedMallocPool.hpp
  IOManager.hpp
  Logger.hpp
  MPI.hpp
  Macros.hpp
  MemoryResourceTraits.hpp
  MemoryMap.hpp
  MemoryMap.inl
  OutputBuffer.hpp
  Platform.hpp
  allocation_statistics.hpp
  detect_vendor.hpp)

if (ENABLE_STATISTICS)
  set (umpire_util_headers
    ${umpire_util_headers}
    statistic_helper.hpp
    Statistic.hpp
    StatisticsDatabase.hpp)
endif ()

if (ENABLE_NUMA)
  set (umpire_util_headers
    ${umpire_util_headers}
    numa.hpp)
endif ()

set (umpire_util_sources
  AllocationMap.cpp
  Exception.cpp
  FixedMallocPool.cpp
  IOManager.cpp
  Logger.cpp
  MPI.cpp
  OutputBuffer.cpp
  allocation_statistics.cpp
  detect_vendor.cpp)

if (ENABLE_STATISTICS)
  set (umpire_util_sources
    ${umpire_util_sources}
    Statistic.cpp
    StatisticsDatabase.cpp)
endif ()

if (ENABLE_NUMA)
  set (umpire_util_sources
    ${umpire_util_sources}
    numa.cpp)
endif ()

set (umpire_util_depends umpire_tpl_judy umpire_tpl_backtrace)

if (ENABLE_NUMA)
  set (umpire_util_depends
    ${umpire_util_depends}
    numa)
endif ()

if (ENABLE_STATISTICS)
  set (umpire_util_depends
    ${umpire_util_depends}
    conduit)
endif ()

if (ENABLE_SLIC AND ENABLE_LOGGING)
  set (umpire_util_depends
    ${umpire_util_depends}
    slic)
endif()

if (ENABLE_MPI)
  set (umpire_util_depends
    ${umpire_util_depends}
    mpi)
endif ()

blt_add_library(
  NAME umpire_util
  HEADERS ${umpire_util_headers}
  SOURCES ${umpire_util_sources}
  DEPENDS_ON ${umpire_util_depends}
  OBJECT TRUE)

target_include_directories(
  umpire_util
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(FILES
  ${umpire_util_headers}
  DESTINATION include/umpire/util)
