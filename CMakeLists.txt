##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

cmake_policy(SET CMP0025 NEW)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0057 NEW)

project(Umpire
  LANGUAGES CXX C
  VERSION 4.1.0)

set(UMPIRE_VERSION_RC "")

option(ENABLE_DEVELOPER_DEFAULTS "Enable default options for Umpire developers" Off)
mark_as_advanced(ENABLE_DEVELOPER_DEFAULTS)

if (ENABLE_DEVELOPER_DEFAULTS)
  set(ENABLE_WARNINGS_AS_ERRORS On CACHE BOOL "")
  set(ENABLE_TOOLS On CACHE BOOL "")
  set(ENABLE_DEVICE_CONST On CACHE BOOL "")
endif ()

set(ENABLE_CUDA Off CACHE BOOL "")
set(ENABLE_HIP Off CACHE BOOL "")
set(ENABLE_SYCL Off CACHE BOOL "")
set(ENABLE_NUMA Off CACHE BOOL "")
set(ENABLE_OPENMP Off CACHE BOOL "")
set(ENABLE_COPY_HEADERS Off CACHE BOOL "")
set(ENABLE_TESTS On CACHE BOOL "")
set(ENABLE_BENCHMARKS On CACHE BOOL "")
set(ENABLE_GMOCK On CACHE BOOL "")
set(ENABLE_DOCS Off CACHE BOOL "")
set(ENABLE_CLANGQUERY Off CACHE BOOL "")
option(ENABLE_EXAMPLES "Build Umpire examples" On)
option(ENABLE_LOGGING "Build Umpire with Logging enabled" On)
option(ENABLE_SLIC "Build Umpire with SLIC logging" Off)
option(ENABLE_BACKTRACE "Build Umpire with allocation backtrace enabled" Off)
option(ENABLE_BACKTRACE_SYMBOLS "Build Umpire with symbol support" Off)
option(ENABLE_STATISTICS "Track statistics for allocations and operations" Off)
option(ENABLE_COVERAGE "Enable code coverage (with GCC)" Off)
option(ENABLE_PEDANTIC_WARNINGS "Enable pedantic compiler warnings" On)
option(ENABLE_TOOLS "Enable Umpire development tools" Off)
option(ENABLE_DEVICE_CONST "Enable constant memory on GPUs" Off)
option(ENABLE_OPENMP_TARGET "Enable OpenMP target support" Off)
if(WIN32 OR APPLE)
  set(ENABLE_FILE_RESOURCE Off CACHE BOOL "")
endif()
option(ENABLE_FILE_RESOURCE "Enable File Resource" On)

# Set benchmark installation to OFF
set(BENCHMARK_ENABLE_INSTALL OFF)

set(BLT_CXX_STD "c++11" CACHE STRING "Version of C++ standard")

if (ENABLE_CUDA)
  cmake_minimum_required(VERSION 3.9)
else ()
  cmake_minimum_required(VERSION 3.8.2)
endif ()

message(STATUS "Using CMake version ${CMAKE_VERSION}")

if (ENABLE_FORTRAN)
  set(ENABLE_C On)
endif()

if (ENABLE_DOCS AND NOT ENABLE_DOXYGEN)
  message(FATAL_ERROR "\
  Sphinx documentation requires Doxygen, \
  please re-configure with ENABLE_DOXYGEN=ON")
endif ()

if (ENABLE_NUMA AND (NOT UNIX OR APPLE))
  message(FATAL_ERROR "\
  NUMA support unavailable. \
  Please re-configure with ENABLE_NUMA=Off (default value)")
endif ()

if (ENABLE_CUDA)
  if ((CMAKE_CXX_COMPILER_ID MATCHES GNU) AND (CMAKE_SYSTEM_PROCESSOR MATCHES ppc64le))
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
      set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -mno-float128")
    endif ()
  endif ()
endif()

if (ENABLE_SYCL)
   if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
   endif()
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
      message(FATAL_ERROR
          "The BLT git submodule is not present. "
          "Either run the following two commands in your git repository: \n"
          "    git submodule init\n"
          "    git submodule update\n"
          "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
    endif ()
  endif ()

  include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
endif()

if (Git_FOUND)
  blt_git_hashcode (HASHCODE umpire_sha1
                    RETURN_CODE rc
                    SOURCE_DIR ${PROJECT_SOURCE_DIR})
  set (UMPIRE_VERSION_RC ${umpire_sha1})
endif ()

include(cmake/SetupCMakeBasics.cmake)
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
  add_subdirectory(tests)
endif ()
if (ENABLE_BENCHMARKS)
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
