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

set(OLD_ENABLE_TESTS ${UMPIRE_ENABLE_TESTS})
set(OLD_ENABLE_FORTRAN ${UMPIRE_ENABLE_FORTRAN})
set(UMPIRE_ENABLE_TESTS Off)
set(UMPIRE_ENABLE_FORTRAN Off)

#
# Json
set (json_headers
  umpire/json/json.hpp )

blt_add_library(
  NAME umpire_tpl_json
  HEADERS ${json_headers})

if (C_COMPILER_FAMILY_IS_INTEL)
  target_compile_options(
    umpire_tpl_json
    INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-wd186>
    $<$<COMPILE_LANGUAGE:C>:-wd186>
  )
endif ()

if (C_COMPILER_FAMILY_IS_PGI)
  target_compile_options(
    umpire_tpl_json
    INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:--diag_suppress 186>
  )
endif ()

target_link_libraries(umpire_tpl_json INTERFACE)

target_include_directories(
  umpire_tpl_json
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/tpl>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)

blt_convert_to_system_includes(TARGET umpire_tpl_json)

target_include_directories(
  umpire_tpl_json
  INTERFACE
  $<INSTALL_INTERFACE:include>)

install(FILES
  ${json_headers}
  DESTINATION include/umpire/json)

install(TARGETS
  umpire_tpl_json
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)


#
# CLI11 Option Parsing Headers
set (CLI11_headers
  umpire/CLI11/CLI11.hpp )

blt_add_library(
  NAME umpire_tpl_CLI11
  HEADERS ${CLI11_headers})

target_link_libraries(umpire_tpl_CLI11 INTERFACE)

target_include_directories(
  umpire_tpl_CLI11
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/tpl>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)

blt_convert_to_system_includes(TARGET umpire_tpl_CLI11)

target_include_directories(
  umpire_tpl_CLI11
  INTERFACE
  $<INSTALL_INTERFACE:include>)

install(FILES
  ${CLI11_headers}
  DESTINATION include/umpire/CLI11)

install(TARGETS
  umpire_tpl_CLI11
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

add_subdirectory(umpire/judy)

if (NOT TARGET camp)
  if (DEFINED camp_DIR)
    find_package(camp REQUIRED
      NO_DEFAULT_PATH
      PATHS ${camp_DIR}
      ${camp_DIR}/lib/cmake/camp
    )
    set_target_properties(camp PROPERTIES IMPORTED_GLOBAL TRUE)
  else ()
    add_subdirectory(umpire/camp)
  endif()

  if(UMPIRE_ENABLE_CUDA)
    blt_add_target_definitions(
      TO camp
      SCOPE INTERFACE
      TARGET_DEFINITIONS CAMP_HAVE_CUDA)
  endif()

  if(ENABLE_HIP)
    blt_add_target_definitions(
      TO camp
      SCOPE INTERFACE
      TARGET_DEFINITIONS CAMP_HAVE_HIP)
  endif()
endif ()

if (NOT TARGET fmt::fmt)
  if (DEFINED fmt_DIR)
    find_package(fmt 9.1...10 REQUIRED NO_DEFAULT_PATH PATHS ${fmt_DIR})
  else ()
    set(FMT_INSTALL ON)
    set(FMT_SYSTEM_HEADERS ON)
    add_subdirectory(umpire/fmt)
  endif ()
endif ()

set(UMPIRE_ENABLE_TESTS ${OLD_ENABLE_TESTS})
set(UMPIRE_ENABLE_FORTRAN ${OLD_ENABLE_FORTRAN})
