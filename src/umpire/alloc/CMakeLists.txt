##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set(umpire_alloc_headers
  malloc_allocator.hpp)

if (ENABLE_NUMA)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    posix_memalign_allocator.hpp)
endif ()

if (ENABLE_CUDA)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    cuda_malloc_allocator.hpp
    cuda_malloc_managed_allocator.hpp
    cuda_pinned_allocator.hpp)
endif ()

if (ENABLE_HIP)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    hip_malloc_allocator.hpp
    hip_pinned_allocator.hpp)
endif ()

if (ENABLE_OPENMP_TARGET)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    omp_target_allocator.hpp)
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
