##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

add_subdirectory(util)
add_subdirectory(resource)
add_subdirectory(alloc)
add_subdirectory(event)
add_subdirectory(op)
add_subdirectory(strategy)
if (SHROUD_FOUND)
  add_subdirectory(interface)
endif ()
if (UMPIRE_ENABLE_C OR UMPIRE_ENABLE_FORTRAN)
  add_subdirectory(interface/c_fortran)
endif()

set (umpire_headers
  Allocator.hpp
  Allocator.inl
  ResourceManager.hpp
  ResourceManager.inl
  Tracking.hpp
  TypedAllocator.hpp
  TypedAllocator.inl
  Umpire.hpp)

set (umpire_sources
  Allocator.cpp
  detail/registry.cpp
  memory.cpp
  ResourceManager.cpp
  Umpire.cpp)

set(umpire_api_v2_headers)

if (UMPIRE_ENABLE_API_V2)
  set(umpire_api_v2_headers
    ${PROJECT_SOURCE_DIR}/include/umpire/allocation_record.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/allocator.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/error.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/memory.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/memory_resource.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/op.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/platform.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/detail/registry.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/op/copy.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/op/dispatch.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/op/host.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/op/memset.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/op/openmp_target.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/op/operations.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/op/prefetch.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/op/reallocate.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/resource/host_memory.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/resource/null_resource.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/resource/platform.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/strategy/allocation_strategy.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/strategy/dynamic_pool_list.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/strategy/fixed_pool.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/strategy/monotonic_buffer.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/strategy/named.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/strategy/quick_pool.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/strategy/size_limiter.hpp
    ${PROJECT_SOURCE_DIR}/include/umpire/strategy/thread_safe.hpp)

  if (UMPIRE_ENABLE_CUDA)
    list(APPEND umpire_api_v2_headers
      ${PROJECT_SOURCE_DIR}/include/umpire/op/cuda.hpp
      ${PROJECT_SOURCE_DIR}/include/umpire/resource/cuda_device_memory.hpp)
  endif ()

  if (UMPIRE_ENABLE_HIP)
    list(APPEND umpire_api_v2_headers
      ${PROJECT_SOURCE_DIR}/include/umpire/op/hip.hpp
      ${PROJECT_SOURCE_DIR}/include/umpire/resource/hip_device_memory.hpp)
  endif ()

  if (UMPIRE_ENABLE_SYCL)
    list(APPEND umpire_api_v2_headers
      ${PROJECT_SOURCE_DIR}/include/umpire/op/sycl.hpp
      ${PROJECT_SOURCE_DIR}/include/umpire/resource/sycl_device_memory.hpp)
  endif ()

  if (UMPIRE_ENABLE_OPENMP_TARGET)
    list(APPEND umpire_api_v2_headers
      ${PROJECT_SOURCE_DIR}/include/umpire/resource/openmp_target_memory.hpp)
  endif ()

  list(APPEND umpire_sources api_v2_instantiations.cpp)
endif ()

if (UMPIRE_ENABLE_DEVICE_ALLOCATOR AND (UMPIRE_ENABLE_CUDA OR UMPIRE_ENABLE_HIP))
  set(umpire_device_headers
    DeviceAllocator.hpp
    device_allocator_helper.hpp)

  set(umpire_device_sources
    DeviceAllocator.cpp
    device_allocator_helper.cpp)

  if (UMPIRE_ENABLE_CUDA)
    set_source_files_properties(DeviceAllocator.cpp
                              PROPERTIES
                              LANGUAGE CUDA)
    set_source_files_properties(device_allocator_helper.cpp
                              PROPERTIES
                              LANGUAGE CUDA)
    set(umpire_device_depends camp cuda umpire)
  elseif (UMPIRE_ENABLE_HIP)
    set(umpire_device_depends camp blt::hip umpire)
    set(hip_version_min "5.4.0")

    if (${hip_VERSION} VERSION_LESS ${hip_version_min})
      message(FATAL_ERROR "ROCM version must be ${hip_version_min} or higher for HIP DeviceAllocator functionality!")
    endif ()
  endif()

  blt_add_library(
    NAME umpire_device
    HEADERS ${umpire_device_headers}
    SOURCES ${umpire_device_sources}
    DEPENDS_ON ${umpire_device_depends})

  target_include_directories(
    umpire_device
    PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

  install(TARGETS
    umpire_device
    EXPORT umpire-targets
    RUNTIME DESTINATION lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

  install(FILES
    ${umpire_device_headers}
    DESTINATION include/umpire)

  if (UMPIRE_ENABLE_CUDA)
    set_target_properties(umpire_device PROPERTIES
      CUDA_SEPARABLE_COMPILATION On)
  else ()
    target_link_options(umpire_device PUBLIC -fgpu-rdc)
    target_compile_options(umpire_device PUBLIC -fgpu-rdc)
  endif ()

endif()

if (UMPIRE_ENABLE_SYCL)
  set(umpire_headers
      ${umpire_headers}
      util/sycl_compat.hpp)
endif()

set (umpire_depends
  camp
  umpire_resource
  umpire_strategy
  umpire_alloc
  umpire_op
  umpire_util
  umpire_event
  umpire_tpl_judy)

if (UMPIRE_ENABLE_HIP)
  set (umpire_depends
    ${umpire_depends}
    blt::hip_runtime)
endif()

if (UMPIRE_ENABLE_OPENMP_TARGET)
  set (umpire_depends
    ${umpire_depends}
    umpire_event
    openmp)
endif ()

blt_list_append( TO umpire_depends ELEMENTS umpire_interface IF UMPIRE_ENABLE_C )

set (umpire_defines)

if (COMPILER_FAMILY_IS_MSVC AND NOT BUILD_SHARED_LIBS)
  set (umpire_defines
    ${umpire_defines}
    UMPIRE_WIN_STATIC_BUILD)
endif ()

blt_add_library(
  NAME umpire
  HEADERS ${umpire_headers}
  SOURCES ${umpire_sources}
  DEPENDS_ON ${umpire_depends}
  DEFINES ${umpire_defines})

if (UMPIRE_ENABLE_API_V2)
  target_sources(umpire PRIVATE ${umpire_api_v2_headers})
  target_compile_definitions(umpire PUBLIC UMPIRE_ENABLE_API_V2=1)
else ()
  target_compile_definitions(umpire PUBLIC UMPIRE_ENABLE_API_V2=0)
endif ()

if (NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS 3.18)
  target_link_options(
    umpire INTERFACE
    $<BUILD_INTERFACE:$<$<NOT:$<LINK_LANGUAGE:Fortran>>:-rdynamic>>
  )

  target_link_libraries(
    umpire INTERFACE $<BUILD_INTERFACE:${CMAKE_DL_LIBS}>
  )
endif()

if (UMPIRE_ENABLE_IPC_SHARED_MEMORY)
  message(STATUS "Host (IPC) Shared Memory Enabled")
  find_package(Threads)
  target_link_libraries(
    umpire INTERFACE Threads::Threads -lrt 
  )
else()
  message(STATUS "IPC Shared Memory Disabled")
endif()

if (UMPIRE_ENABLE_MPI3_SHARED_MEMORY)
  message(STATUS "Host (MPI3) Shared Memory Enabled")
else()
  message(STATUS "MPI3 Shared Memory Disabled")
endif()

if (UMPIRE_ENABLE_CUDA OR UMPIRE_ENABLE_HIP)
  set(UMPIRE_ENABLE_DEVICE On)
  set(UMPIRE_ENABLE_PINNED On)
  set(UMPIRE_ENABLE_UM On)
  if (UMPIRE_ENABLE_DEVICE_CONST)
    set(UMPIRE_ENABLE_CONST On)
  endif()
elseif (UMPIRE_ENABLE_SYCL)
  set(UMPIRE_ENABLE_DEVICE On)
  set(UMPIRE_ENABLE_UM On)
  set(UMPIRE_ENABLE_PINNED On)
else ()
  set(UMPIRE_ENABLE_DEVICE Off)
  set(UMPIRE_ENABLE_PINNED Off)
  set(UMPIRE_ENABLE_CONST Off)
  set(UMPIRE_ENABLE_UM Off)
endif()

if (UMPIRE_ENABLE_OPENMP_TARGET)
  set(UMPIRE_ENABLE_DEVICE On)
endif ()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in
  ${CMAKE_BINARY_DIR}/include/umpire/config.hpp)

target_include_directories(
  umpire
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS
  umpire
  EXPORT umpire-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES
  ${umpire_headers}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/umpire)

install(FILES
  ${CMAKE_BINARY_DIR}/include/umpire/config.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/umpire)

if (UMPIRE_ENABLE_API_V2)
  install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/umpire/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/umpire
    FILES_MATCHING PATTERN "*.hpp")
endif ()

umpire_add_code_checks(PREFIX umpire)
