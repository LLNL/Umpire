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

cmake_policy(SET CMP0025 NEW)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0057 NEW)

project(Umpire 
  LANGUAGES CXX C
  VERSION 0.3.1)

set(ENABLE_OPENMP On CACHE Bool "")
set(ENABLE_CUDA On CACHE Bool "")
set(ENABLE_COPY_HEADERS Off CACHE Bool "")
set(ENABLE_TESTS On CACHE Bool "")
set(ENABLE_BENCHMARKS On CACHE Bool "")
set(ENABLE_GMOCK On CACHE Bool "")
set(ENABLE_DOCS Off CACHE Bool "")
option(ENABLE_FORTRAN_API "Build FORTRAN API" Off)
option(ENABLE_C_API "Build C API" Off)
set(ENABLE_WARNINGS_AS_ERRORS On CACHE Bool "")
option(ENABLE_EXAMPLES "Build Umpire examples" On)
option(ENABLE_LOGGING "Build Umpire with Logging enabled" On)
option(ENABLE_SLIC "Build Umpire with SLIC logging" Off)
option(ENABLE_ASSERTS "Build Umpire with assert() enabled" On)
set(ENABLE_GTEST_DEATH_TESTS ${ENABLE_ASSERTS} CACHE Bool "")
option(ENABLE_STATISTICS "Track statistics for allocations and operations" Off)
option(ENABLE_COVERAGE "Enable code coverage (with GCC)" Off)
option(ENABLE_TOOLS "Enable Umpire Development Tools" On)

if (ENABLE_CUDA)
  cmake_minimum_required(VERSION 3.9)
else ()
  cmake_minimum_required(VERSION 3.8.2)
endif ()

message(STATUS "Using CMake version ${CMAKE_VERSION}")

if (ENABLE_FORTRAN_API)
  set(ENABLE_FORTRAN On)
  set(ENABLE_C_API On)
endif()

################################
# BLT
################################
if (NOT BLT_LOADED)
  if (DEFINED BLT_SOURCE_DIR)
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
  else ()
    set (BLT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/blt CACHE PATH "")

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
      message(FATAL_ERROR "\
      The BLT submodule is not present. \
      If in git repository run the following two commands:\n \
      git submodule init\n \
      git submodule update")
    endif ()
  endif ()

  include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
endif()


include(cmake/SetupCompilerFlags.cmake)
include(cmake/SetupUmpireThirdParty.cmake)

configure_file(
  umpire-config.cmake.in
  "${PROJECT_BINARY_DIR}/umpire-config.cmake" @ONLY)

install(FILES
  "${PROJECT_BINARY_DIR}/umpire-config.cmake"
  DESTINATION share/umpire/cmake)

install(EXPORT umpire-targets DESTINATION share/umpire/cmake)

add_subdirectory(src)

if (ENABLE_TESTS)
  set (UMPIRE_TEST_OUTPUT_DIR ${CMAKE_BINARY_DIR}/test)
  add_subdirectory(tests)
endif ()
if (ENABLE_BENCHMARKS)
  set (UMPIRE_BENCHMARK_OUTPUT_DIR ${CMAKE_BINARY_DIR}/benchmark)
  add_subdirectory(benchmarks)
endif ()
if (ENABLE_EXAMPLES)
add_subdirectory(examples)
endif ()

if (ENABLE_TOOLS)
add_subdirectory(tools)
endif ()

if (ENABLE_DOCS)
  add_subdirectory(docs)
endif ()
