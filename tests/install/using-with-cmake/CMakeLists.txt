##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

cmake_minimum_required(VERSION 3.23)

project(using_with_cmake)

if(NOT DEFINED umpire_DIR)
    message(FATAL_ERROR "Missing required 'umpire_DIR' variable pointing to an installed Umpire")
endif()

find_package(umpire REQUIRED
             NO_DEFAULT_PATH 
             PATHS ${umpire_DIR})

# Check if fmt::fmt was imported from above find_package
if(NOT TARGET fmt::fmt)
    message(FATAL_ERROR "fmt targets were not imported from installed config files.")
endif()

# Check that the backwards-compatible non-namespaced target exists and is a
# real target, not an ALIAS. Tools such as BLT's
# blt_convert_to_system_includes() modify target properties, which CMake
# forbids on ALIAS targets, so verify a property can be set on it.
if(NOT TARGET umpire)
    message(FATAL_ERROR "Backwards-compatible 'umpire' target was not imported from installed config files.")
endif()

get_target_property(_umpire_aliased umpire ALIASED_TARGET)
if(_umpire_aliased)
    message(FATAL_ERROR "Backwards-compatible 'umpire' target must not be an ALIAS target (aliases ${_umpire_aliased}).")
endif()

get_target_property(_umpire_includes umpire INTERFACE_INCLUDE_DIRECTORIES)
if(NOT _umpire_includes)
    message(FATAL_ERROR "Backwards-compatible 'umpire' target has no INTERFACE_INCLUDE_DIRECTORIES.")
endif()
set_property(TARGET umpire PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_umpire_includes}")

add_executable(using-with-cmake using-with-cmake.cpp)
target_link_libraries(using-with-cmake umpire::umpire fmt::fmt)

# Also build against the deprecated non-namespaced target
add_executable(using-with-cmake-compat using-with-cmake.cpp)
target_link_libraries(using-with-cmake-compat umpire)
