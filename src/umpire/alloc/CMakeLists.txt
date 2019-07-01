##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set(umpire_alloc_headers
  MallocAllocator.hpp)

if (ENABLE_SICM)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    SICMAllocator.hpp)
endif()

if (ENABLE_NUMA)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    PosixMemalignAllocator.hpp)
endif ()

if (ENABLE_CUDA)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    CudaMallocAllocator.hpp
    CudaMallocManagedAllocator.hpp
    CudaPinnedAllocator.hpp)
endif ()

if (ENABLE_HCC)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    AmAllocAllocator.hpp)
endif ()

if (ENABLE_HIP)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    HipMallocAllocator.hpp
    HipPinnedAllocator.hpp)
endif ()

if ("${umpire_alloc_sources}" STREQUAL "")
  blt_add_library(
    NAME umpire_alloc
    HEADERS ${umpire_alloc_headers})

  target_include_directories(
    umpire_alloc
    INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

  install(TARGETS
    umpire_alloc
    EXPORT umpire-targets
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)
else()
  blt_add_library(
    NAME umpire_alloc
    HEADERS ${umpire_alloc_headers}
    SOURCES ${umpire_alloc_sources}
    OBJECT TRUE)

  target_include_directories(
    umpire_alloc
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)
endif()

install(FILES
  ${umpire_alloc_headers}
  DESTINATION include/umpire/alloc)
