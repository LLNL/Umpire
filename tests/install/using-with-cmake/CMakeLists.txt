##############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

cmake_minimum_required(VERSION 3.14)

project(using_with_cmake)

if(NOT DEFINED umpire_DIR OR NOT EXISTS ${umpire_DIR}/lib/cmake/umpire/umpire-config.cmake)
    message(FATAL_ERROR "Missing required 'umpire_DIR' variable pointing to an installed Umpire")
endif()

find_package(umpire REQUIRED
             NO_DEFAULT_PATH 
             PATHS ${umpire_DIR}/lib/cmake/umpire)

add_executable(using-with-cmake using-with-cmake.cpp)
target_link_libraries(using-with-cmake umpire)
