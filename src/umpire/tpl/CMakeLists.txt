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

set(OLD_COPY_HEADERS ${ENABLE_COPY_HEADERS})
set(OLD_ENABLE_TESTS ${ENABLE_TESTS})
set(OLD_ENABLE_FORTRAN ${ENABLE_FORTRAN})
set(ENABLE_COPY_HEADERS Off)
set(ENABLE_TESTS Off)
set(ENABLE_FORTRAN Off)

if (ENABLE_STATISTICS)
  add_subdirectory(conduit/src)
  set (conduit_includes
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/conduit/src/libs>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/conduit/src/libs>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/conduit/src/libs/conduit>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/conduit/src/libs/conduit/c>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/conduit/src/libs/conduit/python>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/conduit/src/libs/conduit>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/conduit/src/libs/conduit/python>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

  blt_register_library(
    NAME conduit
    INCLUDES ${conduit_includes}
    TREAT_INCLUDES_AS_SYSTEM On
    LIBRARIES conduit)
endif()

#
# Json
set (json_headers
  ./json/json.hpp )

blt_add_library(
  NAME umpire_tpl_json
  HEADERS ${json_headers})

if (C_COMPILER_FAMILY_IS_PGI)
  target_compile_options(
    umpire_tpl_json
    INTERFACE
    --diag_suppress186)
endif ()
target_link_libraries(umpire_tpl_json INTERFACE)

install(FILES
  ${json_headers}
  DESTINATION include/umpire/tpl/json)

install(TARGETS
  umpire_tpl_json
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

#
# CXX Option Parsing Headers
set (cxxopts_headers
  ./cxxopts/include/cxxopts.hpp )

blt_add_library(
  NAME umpire_tpl_cxxopts
  HEADERS ${cxxopts_headers})

target_link_libraries(umpire_tpl_cxxopts INTERFACE)

target_include_directories(
  umpire_tpl_cxxopts
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(FILES
  ${cxxopts_headers}
  DESTINATION include/umpire/tpl/cxxopts/include)

install(TARGETS
  umpire_tpl_cxxopts
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

add_subdirectory(judy)

if (NOT TARGET camp)
  if (camp_DIR)
    find_package(camp REQUIRED)

    set(CAMP_INCLUDE_DIRS "${camp_INSTALL_PREFIX}/include")
    set(CAMP_DEPENDS )
    blt_list_append(TO CAMP_DEPENDS ELEMENTS cuda_runtime IF ENABLE_CUDA)

    blt_register_library(
      NAME camp
      INCLUDES ${CAMP_INCLUDE_DIRS}
      DEPENDS_ON ${CAMP_DEPENDS})
  else ()
    add_subdirectory(camp)
  endif()
endif ()

set(ENABLE_COPY_HEADERS ${OLD_COPY_HEADERS})
set(ENABLE_ENABLE_TESTS ${OLD_ENABLE_TESTS})
set(ENABLE_FORTRAN ${OLD_ENABLE_FORTRAN})
