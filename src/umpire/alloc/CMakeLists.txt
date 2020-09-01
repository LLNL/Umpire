##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set(umpire_alloc_headers
  MallocAllocator.hpp)

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

if (ENABLE_HIP)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    HipMallocAllocator.hpp
    HipPinnedAllocator.hpp)
endif ()

if (ENABLE_SYCL)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    SyclMallocAllocator.hpp
    SyclMallocManagedAllocator.hpp
    SyclPinnedAllocator.hpp)
endif ()

if (ENABLE_OPENMP_TARGET)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    OpenMPTargetAllocator.hpp)
endif ()

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

install(FILES
  ${umpire_alloc_headers}
  DESTINATION include/umpire/alloc)

umpire_add_code_checks(PREFIX alloc)
