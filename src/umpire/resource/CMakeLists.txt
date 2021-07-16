##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (umpire_resource_headers host_memory.hpp)
set (umpire_resource_depends camp umpire_alloc umpire_detail)

if (ENABLE_CUDA)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    cuda_device_memory.hpp
    cuda_managed_memory.hpp
    cuda_pinned_memory.hpp
    )

  set(umpire_resource_depends
    ${umpire_resource_depends}
    cuda_runtime)
endif ()

if (ENABLE_HIP)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    hip_device_memory.hpp
    hip_pinned_memory.hpp)

  set(umpire_resource_depends
    ${umpire_resource_depends}
    hip_runtime)
endif ()

if (ENABLE_OPENMP_TARGET)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    omp_target_memory.hpp)

  set (umpire_resource_depends
    ${umpire_resource_depends}
    openmp)
endif ()

blt_add_library(
  NAME umpire_resource
  HEADERS ${umpire_resource_headers})

target_include_directories(
  umpire_resource
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

target_link_libraries(
  umpire_resource
  INTERFACE
  ${umpire_resource_depends}
)

install(TARGETS
  umpire_resource
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES
  ${umpire_resource_headers}
  DESTINATION include/umpire/resource)
