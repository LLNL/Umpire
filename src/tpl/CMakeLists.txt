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

set(fmt_headers
    umpire/fmt/fmt.hpp
    umpire/fmt/args.h
    umpire/fmt/chrono.h
    umpire/fmt/color.h
    umpire/fmt/compile.h
    umpire/fmt/core.h
    umpire/fmt/format-inl.h
    umpire/fmt/format.h
    umpire/fmt/locale.h
    umpire/fmt/os.h
    umpire/fmt/ostream.h
    umpire/fmt/printf.h
    umpire/fmt/ranges.h
    )

blt_add_library(NAME    umpire_tpl_fmt
                HEADERS ${fmt_headers})


target_include_directories(
  umpire_tpl_fmt
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/tpl>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)

# Avoid warnings from fmt (so we can still use -Werror)
blt_convert_to_system_includes(TARGET umpire_tpl_fmt)

target_include_directories(
  umpire_tpl_fmt
  INTERFACE
  $<INSTALL_INTERFACE:include>)

if (C_COMPILER_FAMILY_IS_XL)
  set(_fmt_warning_disable_flag
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-unknown-attributes> 
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-unknown-attributes>)

    target_compile_options(umpire_tpl_fmt INTERFACE ${_fmt_warning_disable_flag})
endif ()

if (C_COMPILER_FAMILY_IS_PGI)
  set(_fmt_warning_disable_flag
   "--diag_suppress 1625;--diag_suppress 185;--diag_suppress 811;--diag_suppress 186")
   
  if (ENABLE_FORTRAN)
    target_compile_options(umpire_tpl_fmt
      INTERFACE 
      $<$<NOT:$<COMPILE_LANGUAGE:Fortran>>:${_fmt_warning_disable_flag}>)
  else ()
    target_compile_options(umpire_tpl_fmt
      INTERFACE 
      ${_fmt_warning_disable_flag})
  endif ()
endif ()

if (C_COMPILER_FAMILY_IS_GNU)
  target_compile_options(umpire_tpl_fmt
    INTERFACE 
    $<$<COMPILE_LANGUAGE:CXX>:-fcompare-debug-second>)
endif ()

install(TARGETS
  umpire_tpl_fmt
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES
  ${fmt_headers}
  DESTINATION include/umpire/fmt)

set(UMPIRE_ENABLE_TESTS ${OLD_ENABLE_TESTS})
set(UMPIRE_ENABLE_FORTRAN ${OLD_ENABLE_FORTRAN})
