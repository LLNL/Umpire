##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (umpire_op_headers
  operations.hpp
  dispatch.hpp
  host.hpp)

set (umpire_op_source)

set (umpire_op_depends camp)

if (ENABLE_NUMA)
  set (umpire_op_headers
    ${umpire_op_headers}
    NumaMoveOperation.hpp)

  set (umpire_op_sources
    ${umpire_op_sources}
    NumaMoveOperation.cpp)

  set (umpire_op_depends
    ${umpire_op_depends}
    umpire_strategy)
endif ()

if (ENABLE_CUDA)
  set (umpire_op_headers
    ${umpire_op_headers}
    CudaAdviseAccessedByOperation.hpp
    CudaAdviseUnsetAccessedByOperation.hpp
    CudaAdvisePreferredLocationOperation.hpp
    CudaAdviseUnsetPreferredLocationOperation.hpp
    CudaAdviseReadMostlyOperation.hpp
    CudaAdviseUnsetReadMostlyOperation.hpp
    CudaCopyOperation.hpp
    CudaCopyFromOperation.hpp
    CudaCopyToOperation.hpp
    CudaMemsetOperation.hpp
    CudaMemPrefetchOperation.hpp)

  set (umpire_op_sources
    ${umpire_op_sources}
    CudaAdviseAccessedByOperation.cpp
    CudaAdviseUnsetAccessedByOperation.cpp
    CudaAdvisePreferredLocationOperation.cpp
    CudaAdviseUnsetPreferredLocationOperation.cpp
    CudaAdviseReadMostlyOperation.cpp
    CudaAdviseUnsetReadMostlyOperation.cpp
    CudaCopyOperation.cpp
    CudaCopyFromOperation.cpp
    CudaCopyToOperation.cpp
    CudaMemsetOperation.cpp
    CudaMemPrefetchOperation.cpp)

  set (umpire_op_depends
    ${umpire_op_depends}
    cuda_runtime)
endif ()

if (ENABLE_HIP)
  set (umpire_op_headers
    ${umpire_op_headers}
    HipCopyOperation.hpp
    HipCopyFromOperation.hpp
    HipCopyToOperation.hpp
    HipMemsetOperation.hpp)

  set (umpire_op_sources
    ${umpire_op_sources}
    HipCopyOperation.cpp
    HipCopyFromOperation.cpp
    HipCopyToOperation.cpp
    HipMemsetOperation.cpp)

  set (umpire_op_depends
    ${umpire_op_depends}
    hip_runtime)
endif ()

if (ENABLE_OPENMP_TARGET)
  message(STATUS "Adding OpenMP Target operations")
  set (umpire_op_headers
    ${umpire_op_headers}
    OpenMPTargetCopyOperation.hpp
    OpenMPTargetMemsetOperation.hpp)

  set (umpire_op_sources
    ${umpire_op_sources}
    OpenMPTargetCopyOperation.cpp
    OpenMPTargetMemsetOperation.cpp)

  set (umpire_op_depends
    ${umpire_op_depends}
    openmp)
endif ()

blt_add_library(
  NAME umpire_op
  HEADERS ${umpire_op_headers})
  # SOURCES ${umpire_op_sources}
  #DEPENDS_ON ${umpire_op_depends}
  #OBJECT TRUE)

target_link_libraries(
  umpire_op
  INTERFACE
  ${umpire_op_depends}
)

target_include_directories(
  umpire_op
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS
  umpire_op
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES
  ${umpire_op_headers}
  DESTINATION include/umpire/op)
