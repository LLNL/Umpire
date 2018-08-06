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
set (umpire_resource_headers
  DefaultMemoryResource.hpp
  DefaultMemoryResource.inl
  HostResourceFactory.hpp
  MemoryResource.hpp
  MemoryResourceFactory.hpp
  MemoryResourceRegistry.hpp
  MemoryResourceTypes.hpp
)

set (umpire_resource_sources
  HostResourceFactory.cpp
  MemoryResource.cpp
  MemoryResourceRegistry.cpp
)

set (umpire_resource_depends
  umpire_alloc umpire_strategy umpire_util)

if (ENABLE_CUDA)
  set (umpire_resource_headers
    ${umpire_resource_headers}
    DeviceConstResourceFactory.hpp
    ConstantMemoryResource.hpp
    DeviceResourceFactory.hpp
    PinnedMemoryResourceFactory.hpp
    UnifiedMemoryResourceFactory.hpp)

  set (umpire_resource_sources
    ${umpire_resource_sources}
    DeviceConstResourceFactory.cpp
    ConstantMemoryResource.cu
    DeviceResourceFactory.cpp
    PinnedMemoryResourceFactory.cpp
    UnifiedMemoryResourceFactory.cpp)

  set(umpire_resource_depends
    ${umpire_resource_depends}
    cuda_runtime)
endif ()

blt_add_library(
  NAME umpire_resource
  HEADERS ${umpire_resource_headers}
  SOURCES ${umpire_resource_sources}
  HEADERS_OUTPUT_SUBDIR umpire/resource
  DEPENDS_ON ${umpire_resource_depends})

target_include_directories(
  umpire_resource
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS
  umpire_resource
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES
  ${umpire_resource_headers}
  DESTINATION include/umpire/resource)
