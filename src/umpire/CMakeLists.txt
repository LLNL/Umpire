##############################################################################
# Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
set (umpire_headers
  Allocator.hpp
  Replay.hpp
  ResourceManager.hpp
  ResourceManager.inl
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
    interface/c_fortran/shroudrt.hpp
    interface/c_fortran/wrapAllocator.h
    interface/c_fortran/wrapResourceManager.h
    interface/c_fortran/typesUmpire.h)

  if (ENABLE_C)
    set (umpire_interface_sources
      interface/c_fortran/shroudrt.cpp
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

add_subdirectory(tpl)

add_subdirectory(resource)
add_subdirectory(alloc)
add_subdirectory(op)
add_subdirectory(util)
add_subdirectory(strategy)
if (SHROUD_FOUND)
  add_subdirectory(interface)
endif ()

blt_add_library(
  NAME umpire
  HEADERS ${umpire_headers} ${umpire_interface_headers} ${umpire_interface_c_fortran_headers}
  SOURCES ${umpire_sources} ${umpire_interface_sources}
  DEPENDS_ON umpire_resource umpire_strategy umpire_alloc umpire_op umpire_util umpire_tpl_judy umpire_tpl_simpool)

set(UMPIRE_ENABLE_CUDA ${ENABLE_CUDA})
set(UMPIRE_ENABLE_NUMA ${ENABLE_NUMA})
set(UMPIRE_ENABLE_LOGGING ${ENABLE_LOGGING})
set(UMPIRE_ENABLE_SLIC ${ENABLE_SLIC})
set(UMPIRE_ENABLE_STATISTICS ${ENABLE_STATISTICS})
set(UMPIRE_ENABLE_HCC ${ENABLE_HCC})

if (ENABLE_CUDA OR ENABLE_HCC)
  set(UMPIRE_ENABLE_DEVICE On)
  set(UMPIRE_ENABLE_PINNED On)
else ()
  set(UMPIRE_ENABLE_DEVICE Off)
  set(UMPIRE_ENABLE_PINNED Off)
endif()

if (ENABLE_CUDA)
  set(UMPIRE_ENABLE_UM On)
else ()
  set(UMPIRE_ENABLE_UM Off)
endif()

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
  ${CMAKE_BINARY_DIR}/include
  DESTINATION ${CMAKE_INSTALL_PREFIX})
