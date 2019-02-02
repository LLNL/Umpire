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
set (umpire_op_headers
  GenericReallocateOperation.hpp
  HostCopyOperation.hpp
  HostMemsetOperation.hpp
  HostReallocateOperation.hpp
  MemoryOperation.hpp
  MemoryOperationRegistry.hpp)

set (umpire_op_sources
  GenericReallocateOperation.cpp
  HostCopyOperation.cpp
  HostMemsetOperation.cpp
  HostReallocateOperation.cpp
  MemoryOperation.cpp
  MemoryOperationRegistry.cpp)

set (umpire_op_depends
  umpire_util)

if (ENABLE_NUMA_HOST)
  set (umpire_op_headers
    ${umpire_op_headers}
    NumaReallocateOperation.hpp)

  set (umpire_op_sources
    ${umpire_op_sources}
    NumaReallocateOperation.cpp)
endif ()

if (ENABLE_CUDA)
  set (umpire_op_headers
    ${umpire_op_headers}
    CudaAdviseAccessedByOperation.hpp
    CudaAdvisePreferredLocationOperation.hpp
    CudaAdviseReadMostlyOperation.hpp
    CudaCopyOperation.hpp
    CudaCopyFromOperation.hpp
    CudaCopyToOperation.hpp
    CudaMemsetOperation.hpp)

  set (umpire_op_sources
    ${umpire_op_sources}
    CudaAdviseAccessedByOperation.cpp
    CudaAdvisePreferredLocationOperation.cpp
    CudaAdviseReadMostlyOperation.cpp
    CudaCopyOperation.cpp
    CudaCopyFromOperation.cpp
    CudaCopyToOperation.cpp
    CudaMemsetOperation.cpp)

  set (umpire_op_depends
    ${umpire_op_depends}
    cuda_runtime)
endif ()

if (ENABLE_ROCM)
  set (umpire_op_headers
    ${umpire_op_headers}
    RocmCopyOperation.hpp
    RocmMemsetOperation.hpp)

  set (umpire_op_sources
    ${umpire_op_sources}
    RocmCopyOperation.cpp
    RocmMemsetOperation.cpp)

  set (umpire_op_depends
    ${umpire_op_depends}
    rocm)
endif ()


blt_add_library(
  NAME umpire_op
  HEADERS ${umpire_op_headers}
  SOURCES ${umpire_op_sources}
  HEADERS_OUTPUT_SUBDIR umpire/op
  DEPENDS_ON ${umpire_op_depends}
)

target_include_directories(
  umpire_op
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(FILES
  ${umpire_op_headers}
  DESTINATION include/umpire/op)

install(TARGETS
  umpire_op
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
