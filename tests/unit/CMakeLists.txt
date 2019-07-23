##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (unit_dir ${UMPIRE_TESTS_OUTPUT_DIR}/unit)

set (unit_depends umpire gtest)

if (ENABLE_HIP)
  set (unit_depends ${unit_depends} hip_runtime)
endif()

blt_add_executable(
  NAME resource_manager_tests
  SOURCES resource_manager_tests.cpp
  DEPENDS_ON ${unit_depends}
  OUTPUT_DIR ${unit_dir})

blt_add_test(
  NAME resource_manager_tests
  COMMAND resource_manager_tests)

blt_add_executable(
  NAME version_tests
  SOURCES version_tests.cpp
  DEPENDS_ON ${unit_depends}
  OUTPUT_DIR ${unit_dir})

blt_add_test(
  NAME version_tests
  COMMAND version_tests)

add_subdirectory(alloc)
add_subdirectory(op)
add_subdirectory(resource)
add_subdirectory(util)
add_subdirectory(strategy)
