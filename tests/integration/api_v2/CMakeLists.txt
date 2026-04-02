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

blt_add_executable(
  NAME api_v2_operations_headers_tests
  SOURCES test_operations_headers.cpp
  DEPENDS_ON ${integration_tests_depends})

blt_add_test(
  NAME api_v2_operations_headers_tests
  COMMAND api_v2_operations_headers_tests)

blt_add_executable(
  NAME api_v2_operations_tests
  SOURCES test_operations.cpp
  DEPENDS_ON ${integration_tests_depends})

blt_add_test(
  NAME api_v2_operations_tests
  COMMAND api_v2_operations_tests)

blt_add_executable(
  NAME api_v2_v1_interop_tests
  SOURCES test_v1_v2_interop.cpp
  DEPENDS_ON ${integration_tests_depends})

target_include_directories(
  api_v2_v1_interop_tests
  BEFORE PRIVATE
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/src/umpire
  ${PROJECT_SOURCE_DIR}/include)

target_include_directories(
  api_v2_v1_interop_tests
  PRIVATE
  ${PROJECT_SOURCE_DIR}/include)
target_include_directories(
  api_v2_v1_interop_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME api_v2_v1_interop_tests
  COMMAND api_v2_v1_interop_tests)
