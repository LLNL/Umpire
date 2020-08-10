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

if(ENABLE_FILE_RESOURCE)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    FileMemoryResource.hpp
    FileMemoryResourceFactory.hpp
  )
endif()

set (umpire_resource_sources
  HostResourceFactory.cpp
  MemoryResource.cpp
  MemoryResourceRegistry.cpp
  NullMemoryResource.cpp
  NullMemoryResourceFactory.cpp
)

if(ENABLE_FILE_RESOURCE)
  set (umpire_resource_sources
    ${umpire_resource_sources}
    FileMemoryResource.cpp
    FileMemoryResourceFactory.cpp
  )
endif()

set(umpire_resource_depends
  camp)

if (ENABLE_CUDA)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    CudaDeviceMemoryResource.hpp
    CudaDeviceResourceFactory.hpp
    CudaPinnedMemoryResourceFactory.hpp
    CudaUnifiedMemoryResourceFactory.hpp
    )

  set (umpire_resource_sources
    ${umpire_resource_sources}
    CudaDeviceMemoryResource.cpp
    CudaDeviceResourceFactory.cpp
    CudaPinnedMemoryResourceFactory.cpp
    CudaUnifiedMemoryResourceFactory.cpp)

  if (ENABLE_DEVICE_CONST)
    set (umpire_resource_headers
      ${umpire_resource_headers}
      CudaConstantMemoryResource.hpp
      CudaConstantMemoryResourceFactory.hpp)

    set (umpire_resource_sources
      ${umpire_resource_sources}
      CudaConstantMemoryResource.cu
      CudaConstantMemoryResourceFactory.cpp)
  endif ()

  set(umpire_resource_depends
    ${umpire_resource_depends}
    cuda_runtime)
endif ()

if (ENABLE_HIP)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    HipDeviceResourceFactory.hpp
    HipPinnedMemoryResourceFactory.hpp)

  set (umpire_resource_sources
    ${umpire_resource_sources}
    HipDeviceResourceFactory.cpp
    HipPinnedMemoryResourceFactory.cpp)

  set(umpire_resource_depends
    ${umpire_resource_depends}
    hip_runtime)

  if (ENABLE_DEVICE_CONST)
    set (umpire_resource_headers
      ${umpire_resource_headers}
      HipConstantMemoryResourceFactory.hpp
      HipConstantMemoryResource.hpp)

    set (umpire_resource_sources
      ${umpire_resource_sources}
      HipConstantMemoryResourceFactory.cpp
      HipConstantMemoryResource.cpp)
  endif ()
endif ()

if (ENABLE_SYCL)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    SyclDeviceMemoryResource.hpp
    SyclDeviceMemoryResource.inl
    SyclDeviceResourceFactory.hpp
    SyclPinnedMemoryResourceFactory.hpp
    SyclUnifiedMemoryResourceFactory.hpp
    )

  set (umpire_resource_sources
    ${umpire_resource_sources}
    SyclDeviceResourceFactory.cpp
    SyclPinnedMemoryResourceFactory.cpp
    SyclUnifiedMemoryResourceFactory.cpp
    )

  set(umpire_resource_depends
    ${umpire_resource_depends})
endif ()

if (ENABLE_OPENMP_TARGET)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    OpenMPTargetMemoryResourceFactory.hpp)

  set (umpire_resource_sources
    ${umpire_resource_sources}
    OpenMPTargetMemoryResourceFactory.cpp)

  set (umpire_resource_depends
    ${umpire_resource_depends}
    openmp)
endif ()

if (ENABLE_CALIPER)
  set( umpire_resource_depends
    ${umpire_resource_depends}
    cali)
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
