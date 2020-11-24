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

set(OLD_ENABLE_TESTS ${ENABLE_TESTS})
set(OLD_ENABLE_FORTRAN ${ENABLE_FORTRAN})
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
# CLI11 Option Parsing Headers
set (CLI11_headers
  ./CLI11/CLI11.hpp )

blt_add_library(
  NAME umpire_tpl_CLI11
  HEADERS ${CLI11_headers})

target_link_libraries(umpire_tpl_CLI11 INTERFACE)

target_include_directories(
  umpire_tpl_CLI11
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(FILES
  ${CLI11_headers}
  DESTINATION include/umpire/tpl/CLI11)

install(TARGETS
  umpire_tpl_CLI11
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

add_subdirectory(judy)

if (NOT TARGET camp)
  if (DEFINED camp_DIR)
    find_package(camp REQUIRED)
    set_target_properties(camp PROPERTIES IMPORTED_GLOBAL TRUE)
  else ()
    add_subdirectory(camp)
  endif()

  if(ENABLE_CUDA)
    blt_add_target_definitions(
      TO camp
      SCOPE INTERFACE
      TARGET_DEFINITIONS CAMP_HAVE_CUDA)
  endif()

endif ()

set(ENABLE_ENABLE_TESTS ${OLD_ENABLE_TESTS})
set(ENABLE_FORTRAN ${OLD_ENABLE_FORTRAN})
