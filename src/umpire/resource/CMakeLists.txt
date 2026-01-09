##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

set(umpire_resource_depends umpire_util camp ${UMPIRE_FMT_TARGET})

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

if(UMPIRE_ENABLE_DEVELOPER_BENCHMARKS)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    NoOpMemoryResource.hpp
    NoOpResourceFactory.hpp
  )
endif()

set (umpire_resource_sources
  HostResourceFactory.cpp
  MemoryResource.cpp
  MemoryResourceRegistry.cpp
  NullMemoryResource.cpp
  NullMemoryResourceFactory.cpp
)

if(UMPIRE_ENABLE_FILE_RESOURCE)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    FileMemoryResource.hpp
    FileMemoryResourceFactory.hpp
  )
  set (umpire_resource_sources
    ${umpire_resource_sources}
    FileMemoryResource.cpp
    FileMemoryResourceFactory.cpp
  )
  if(UMPIRE_ENABLE_UMAP)
    set (umpire_resource_depends ${umpire_resource_depends} umap)
  endif()
endif()

if(UMPIRE_ENABLE_DEVELOPER_BENCHMARKS)
  set (umpire_resource_sources
    ${umpire_resource_sources}
    NoOpMemoryResource.cpp
    NoOpResourceFactory.cpp
  )
endif()

if(UMPIRE_ENABLE_IPC_SHARED_MEMORY)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    HostSharedMemoryResource.hpp
    HostSharedMemoryResourceFactory.hpp
    HostSharedMemoryResourceImpl.hpp
  )

  set (umpire_resource_sources
    ${umpire_resource_sources}
    HostSharedMemoryResource.cpp
    HostSharedMemoryResourceFactory.cpp
  )
endif()

if(UMPIRE_ENABLE_MPI3_SHARED_MEMORY)
  set( umpire_resource_headers
    ${umpire_resource_headers}
    HostMpi3SharedMemoryResource.hpp
    HostMpi3SharedMemoryResourceFactory.hpp)

  set( umpire_resource_sources
    ${umpire_resource_sources}
    HostMpi3SharedMemoryResource.cpp
    HostMpi3SharedMemoryResourceFactory.cpp)

  set( umpire_resource_depends
  ${umpire_resource_depends}
    mpi)
endif()

if (UMPIRE_ENABLE_CUDA)
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

  if (UMPIRE_ENABLE_DEVICE_CONST)
    set (umpire_resource_headers
      ${umpire_resource_headers}
      CudaConstantMemoryResource.hpp
      CudaConstantMemoryResourceFactory.hpp)

    set (umpire_resource_sources
      ${umpire_resource_sources}
      CudaConstantMemoryResource.cu
      CudaConstantMemoryResourceFactory.cpp)
  endif ()

  set(umpire_resource_depends ${umpire_resource_depends} cuda_runtime)
endif ()

if (UMPIRE_ENABLE_HIP)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    HipDeviceMemoryResource.hpp
    HipDeviceResourceFactory.hpp
    HipUnifiedMemoryResource.hpp
    HipUnifiedMemoryResourceFactory.hpp
    HipPinnedMemoryResource.hpp
    HipPinnedMemoryResourceFactory.hpp)

  set (umpire_resource_sources
    ${umpire_resource_sources}
    HipDeviceMemoryResource.cpp
    HipDeviceResourceFactory.cpp
    HipUnifiedMemoryResource.cpp
    HipUnifiedMemoryResourceFactory.cpp
    HipPinnedMemoryResource.cpp
    HipPinnedMemoryResourceFactory.cpp)

  if (UMPIRE_ENABLE_DEVICE_CONST)
    set (umpire_resource_headers
      ${umpire_resource_headers}
      HipConstantMemoryResourceFactory.hpp
      HipConstantMemoryResource.hpp)

    set (umpire_resource_sources
      ${umpire_resource_sources}
      HipConstantMemoryResourceFactory.cpp
      HipConstantMemoryResource.cpp)

    set_source_files_properties(HipConstantMemoryResource.cpp PROPERTIES LANGUAGE HIP)
  endif ()

  set(umpire_resource_depends ${umpire_resource_depends} blt::hip_runtime)
endif ()

if (UMPIRE_ENABLE_SYCL)
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
endif ()

if (UMPIRE_ENABLE_OPENMP_TARGET)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    OpenMPTargetMemoryResourceFactory.hpp)

  set (umpire_resource_sources
    ${umpire_resource_sources}
    OpenMPTargetMemoryResourceFactory.cpp)

  set (umpire_resource_depends ${umpire_resource_depends} openmp)
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
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(FILES
  ${umpire_resource_headers}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/umpire/resource)

umpire_add_code_checks(PREFIX resource)
