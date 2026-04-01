##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
blt_add_executable(
  NAME api_v2_stl_compatibility_tests
  SOURCES test_stl_compatibility.cpp
  DEPENDS_ON ${integration_tests_depends})

target_include_directories(
  api_v2_stl_compatibility_tests
  BEFORE PRIVATE
  ${PROJECT_SOURCE_DIR}/include)

target_include_directories(
  api_v2_stl_compatibility_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME api_v2_stl_compatibility_tests
  COMMAND api_v2_stl_compatibility_tests)
