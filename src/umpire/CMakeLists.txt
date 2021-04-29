##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (umpire_headers
  Allocator.hpp
  Allocator.inl
  Replay.hpp
  ResourceManager.hpp
  ResourceManager.inl
  Tracking.hpp
  TypedAllocator.hpp
  TypedAllocator.inl
  Umpire.hpp)

set (umpire_sources
  Allocator.cpp
  Replay.cpp
  ResourceManager.cpp
  Umpire.cpp)

if (ENABLE_FORTRAN OR ENABLE_C)
  set (umpire_interface_headers
    interface/umpire.h)

  set (umpire_interface_c_fortran_headers
    interface/c_fortran/wrapAllocator.h
    interface/c_fortran/wrapResourceManager.h
    interface/c_fortran/wrapUmpire.h
    interface/c_fortran/typesUmpire.h)

  if (ENABLE_C)
    set (umpire_interface_sources
      interface/c_fortran/utilUmpire.cpp
      interface/c_fortran/wrapAllocator.cpp
      interface/c_fortran/wrapResourceManager.cpp
      interface/c_fortran/wrapUmpire.cpp)

    if (ENABLE_FORTRAN)
      set (umpire_interface_sources
        ${umpire_interface_sources}
        interface/c_fortran/wrapfumpire.f)

      set_source_files_properties(
        interface/c_fortran/wrapfumpire.f
        PROPERTIES
        Fortran_FORMAT FREE)
    endif ()
  endif ()
endif ()

if (ENABLE_CUDA)
  set(umpire_headers
    ${umpire_headers}
    DeviceAllocator.hpp)

  set(umpire_sources
    ${umpire_sources}
    DeviceAllocator.cpp)

  set_source_files_properties(DeviceAllocator.cpp
                              PROPERTIES
                              LANGUAGE CUDA)
endif()

add_subdirectory(tpl)

add_subdirectory(resource)
add_subdirectory(alloc)
add_subdirectory(op)
add_subdirectory(util)
add_subdirectory(strategy)
if (SHROUD_FOUND)
  add_subdirectory(interface)
endif ()

set (umpire_depends
  camp
  umpire_resource
  umpire_strategy
  umpire_alloc
  umpire_op
  umpire_util
  umpire_tpl_judy)

if (ENABLE_OPENMP_TARGET)
  set (umpire_depends
    ${umpire_depends}
    openmp)
endif ()

blt_list_append( TO umpire_depends ELEMENTS cuda IF ENABLE_CUDA )

set (umpire_defines)

if (COMPILER_FAMILY_IS_MSVC AND NOT BUILD_SHARED_LIBS)
  set (umpire_defines
    ${umpire_defines}
    UMPIRE_WIN_STATIC_BUILD)
endif ()

blt_add_library(
  NAME umpire
  HEADERS ${umpire_headers} ${umpire_interface_headers} ${umpire_interface_c_fortran_headers}
  SOURCES ${umpire_sources} ${umpire_interface_sources}
  DEPENDS_ON ${umpire_depends}
  DEFINES ${umpire_defines})

if (NOT WIN32 AND NOT ${CMAKE_VERSION} VERSION_LESS 3.18)
  target_link_options(
    umpire INTERFACE
    $<BUILD_INTERFACE:$<$<NOT:$<LINK_LANGUAGE:Fortran>>:-rdynamic>>
  )

  target_link_libraries(
    umpire INTERFACE $<BUILD_INTERFACE:${CMAKE_DL_LIBS}>
  )
endif()

set_target_properties(umpire PROPERTIES
  CUDA_SEPARABLE_COMPILATION On)

set(UMPIRE_ENABLE_CUDA ${ENABLE_CUDA})
set(UMPIRE_ENABLE_NUMA ${ENABLE_NUMA})
set(UMPIRE_ENABLE_LOGGING ${ENABLE_LOGGING})
set(UMPIRE_ENABLE_INACCESSIBILITY_TESTS ${ENABLE_INACCESSIBILITY_TESTS})
set(UMPIRE_ENABLE_SLIC ${ENABLE_SLIC})
set(UMPIRE_ENABLE_BACKTRACE ${ENABLE_BACKTRACE})
set(UMPIRE_ENABLE_BACKTRACE_SYMBOLS ${ENABLE_BACKTRACE_SYMBOLS})
set(UMPIRE_ENABLE_DEVELOPER_BENCHMARKS ${ENABLE_DEVELOPER_BENCHMARKS})
set(UMPIRE_ENABLE_MPI ${ENABLE_MPI})
set(UMPIRE_ENABLE_HIP ${ENABLE_HIP})
set(UMPIRE_ENABLE_SYCL ${ENABLE_SYCL})
set(UMPIRE_ENABLE_OPENMP_TARGET ${ENABLE_OPENMP_TARGET})
set(UMPIRE_ENABLE_FILE_RESOURCE ${ENABLE_FILE_RESOURCE})

if (ENABLE_CUDA OR ENABLE_HIP)
  set(UMPIRE_ENABLE_DEVICE On)
  set(UMPIRE_ENABLE_PINNED On)
  set(UMPIRE_ENABLE_UM On)
  if (ENABLE_DEVICE_CONST)
    set(UMPIRE_ENABLE_CONST On)
  endif()
elseif (ENABLE_SYCL)
  set(UMPIRE_ENABLE_DEVICE On)
  set(UMPIRE_ENABLE_UM On)
  set(UMPIRE_ENABLE_PINNED On)
else ()
  set(UMPIRE_ENABLE_DEVICE Off)
  set(UMPIRE_ENABLE_PINNED Off)
  set(UMPIRE_ENABLE_CONST Off)
  set(UMPIRE_ENABLE_UM Off)
endif()

if (ENABLE_ASAN)
  set(UMPIRE_ENABLE_ASAN On)
endif ()

if (ENABLE_OPENMP_TARGET)
  set(UMPIRE_ENABLE_DEVICE On)
endif ()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in
  ${CMAKE_BINARY_DIR}/include/umpire/config.hpp)

target_include_directories(
  umpire
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(TARGETS
  umpire
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES
  ${umpire_headers}
  DESTINATION include/umpire)

install(FILES
  ${umpire_interface_headers}
  DESTINATION include/umpire/interface)

install(FILES
  ${umpire_interface_c_fortran_headers}
  DESTINATION include/umpire/interface/c_fortran)

install(DIRECTORY
  ${CMAKE_Fortran_MODULE_DIRECTORY}/
  DESTINATION include/umpire
  FILES_MATCHING PATTERN "*.mod")

install(DIRECTORY
  ${CMAKE_BINARY_DIR}/include
  DESTINATION ".")

umpire_add_code_checks(PREFIX umpire)
