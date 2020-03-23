##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (umpire_resource_headers
  DefaultMemoryResource.hpp
  DefaultMemoryResource.inl
  HostResourceFactory.hpp
  MemoryResource.hpp
  MemoryResourceFactory.hpp
  MemoryResourceRegistry.hpp
  MemoryResourceTypes.hpp
  NullMemoryResource.hpp
  NullMemoryResourceFactory.hpp
)

set (umpire_resource_sources
  HostResourceFactory.cpp
  MemoryResource.cpp
  MemoryResourceRegistry.cpp
  NullMemoryResource.cpp
  NullMemoryResourceFactory.cpp
)

if (ENABLE_CUDA)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    CudaConstantMemoryResource.hpp
    CudaConstantMemoryResourceFactory.hpp
    CudaDeviceMemoryResource.hpp
    CudaDeviceResourceFactory.hpp
    CudaPinnedMemoryResourceFactory.hpp
    CudaUnifiedMemoryResourceFactory.hpp
    )

  set (umpire_resource_sources
    ${umpire_resource_sources}
    CudaConstantMemoryResource.cu
    CudaConstantMemoryResourceFactory.cpp
    CudaDeviceMemoryResource.cpp
    CudaDeviceResourceFactory.cpp
    CudaPinnedMemoryResourceFactory.cpp
    CudaUnifiedMemoryResourceFactory.cpp)


  set(umpire_resource_depends
    ${umpire_resource_depends}
    cuda_runtime)
endif ()

if (ENABLE_HIP)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    HipDeviceResourceFactory.hpp
    HipPinnedMemoryResourceFactory.hpp
    HipConstantMemoryResourceFactory.hpp
    HipConstantMemoryResource.hpp)

  set (umpire_resource_sources
    ${umpire_resource_sources}
    HipConstantMemoryResourceFactory.cpp
    HipConstantMemoryResource.cpp
    HipDeviceResourceFactory.cpp
    HipPinnedMemoryResourceFactory.cpp)

  set(umpire_resource_depends
    ${umpire_resource_depends}
    hip_runtime)
endif ()

blt_add_library(
  NAME umpire_resource
  HEADERS ${umpire_resource_headers}
  SOURCES ${umpire_resource_sources}
  DEPENDS_ON ${umpire_resource_depends}
  OBJECT TRUE)

target_include_directories(
  umpire_resource
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(FILES
  ${umpire_resource_headers}
  DESTINATION include/umpire/resource)
