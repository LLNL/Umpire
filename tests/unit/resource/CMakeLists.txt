##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (resource_dir ${UMPIRE_TESTS_OUTPUT_DIR}/unit/resource)

set (resource_depends umpire gtest gmock)

if (ENABLE_HIP)
  set (resource_depends ${resource_depends} hip_runtime)
endif()

blt_add_executable(
  NAME null_resource_tests
  SOURCES null_resource_tests.cpp
  DEPENDS_ON ${resource_depends}
  OUTPUT_DIR ${resource_dir})

blt_add_test(
  NAME null_resource_tests
  COMMAND null_resource_tests)


if (ENABLE_GMOCK)
  blt_add_executable(
    NAME resource_tests
    SOURCES resource_tests.cpp
    DEPENDS_ON ${resource_depends}
    OUTPUT_DIR ${resource_dir})

  blt_add_test(
    NAME resource_tests
    COMMAND resource_tests)
endif ()
