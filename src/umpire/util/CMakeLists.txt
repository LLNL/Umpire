##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

set(UMPIRE_ENABLE_SYCL ${UMPIRE_ENABLE_SYCL})

set (umpire_util_headers
  AllocationMap.hpp
  AllocationRecord.hpp
  backtrace.hpp
  backtrace.inl
  error.hpp
  find_first_set.hpp
  FixedMallocPool.hpp
  io.hpp
  Logger.hpp
  MPI.hpp
  Macros.hpp
  MemoryResourceTraits.hpp
  MemoryMap.hpp
  MemoryMap.inl
  OutputBuffer.hpp
  Platform.hpp
  allocation_statistics.hpp
  detect_vendor.hpp
  make_unique.hpp
  memory_sanitizers.hpp
  shared_memory_helper.hpp
  wrap_allocator.hpp)

if (UMPIRE_ENABLE_NUMA)
  set (umpire_util_headers
    ${umpire_util_headers}
    numa.hpp)
endif ()

set (umpire_util_sources
  AllocationMap.cpp
  FixedMallocPool.cpp
  io.cpp
  Logger.cpp
  MPI.cpp
  OutputBuffer.cpp
  allocation_statistics.cpp
  detect_vendor.cpp)

if (UMPIRE_ENABLE_NUMA)
  set (umpire_util_sources
    ${umpire_util_sources}
    numa.cpp)
endif ()

set (umpire_util_depends camp umpire_event umpire_tpl_judy ${UMPIRE_FMT_TARGET})

if (UMPIRE_ENABLE_CUDA OR UMPIRE_ENABLE_HIP)
  set(umpire_device_memset_headers
    device_memset_kernel.hpp)

  set(umpire_device_memset_sources
    device_memset_kernel.cpp)

  if (UMPIRE_ENABLE_CUDA)
    set_source_files_properties(device_memset_kernel.cpp
                              PROPERTIES
                              LANGUAGE CUDA)
    set(umpire_device_memset_depends camp cuda umpire)
  elseif (UMPIRE_ENABLE_HIP)
    set(umpire_device_memset_depends camp blt::hip umpire)
  endif()

  blt_add_library(
    NAME umpire_device_memset
    HEADERS ${umpire_device_memset_headers}
    SOURCES ${umpire_device_memset_sources}
    DEPENDS_ON ${umpire_device_memset_depends})

  target_include_directories(
    umpire_device_memset
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

  install(TARGETS
    umpire_device_memset
    EXPORT umpire-targets
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

  install(FILES
    ${umpire_device_memset_headers}
    DESTINATION include/umpire)

  if (UMPIRE_ENABLE_CUDA)
    set_target_properties(umpire_device_memset PROPERTIES
      CUDA_SEPARABLE_COMPILATION On)
  else ()
    target_link_options(umpire_device_memset PUBLIC -fgpu-rdc)
    target_compile_options(umpire_device_memset PUBLIC -fgpu-rdc)
  endif ()

endif()

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set (umpire_util_depends
    ${umpire_util_depends}
    backtrace_symbols)
endif ()

if (UMPIRE_ENABLE_NUMA)
  set (umpire_util_depends
    ${umpire_util_depends}
    numa)
endif ()

if (UMPIRE_ENABLE_SLIC AND UMPIRE_ENABLE_LOGGING)
  set (umpire_util_depends
    ${umpire_util_depends}
    slic)
endif()

if (UMPIRE_ENABLE_MPI)
  set (umpire_util_depends
    ${umpire_util_depends}
    mpi)
endif ()

if (UMPIRE_ENABLE_OPENMP_TARGET)
  set (umpire_util_depends
    ${umpire_util_depends}
    openmp)
endif ()

if (UMPIRE_ENABLE_HIP)
  set (umpire_util_depends
    ${umpire_util_depends}
    blt::hip_runtime)
endif ()

if (UMPIRE_ENABLE_CUDA)
  set (umpire_util_depends
    ${umpire_util_depends}
    cuda_runtime)
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
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(FILES
  ${umpire_util_headers}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/umpire/util)

umpire_add_code_checks(PREFIX util)
