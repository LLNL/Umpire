##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (util_dir ${UMPIRE_TESTS_OUTPUT_DIR}/unit/util)

blt_add_executable(
  NAME exception_tests
  SOURCES exception_tests.cpp
  DEPENDS_ON umpire gtest
  OUTPUT_DIR ${util_dir})

blt_add_test(
  NAME exception_tests
  COMMAND exception_tests)

blt_add_executable(
  NAME allocation_map_tests
  SOURCES allocation_map_tests.cpp
  DEPENDS_ON umpire gtest
  OUTPUT_DIR ${util_dir})

blt_add_test(
  NAME allocation_map_tests
  COMMAND allocation_map_tests)

blt_add_executable(
  NAME output_buffer_tests
  SOURCES output_buffer_tests.cpp
  DEPENDS_ON umpire gtest
  OUTPUT_DIR ${util_dir})

blt_add_test(
  NAME output_buffer_tests
  COMMAND output_buffer_tests)

blt_add_executable(
  NAME memory_map_tests
  SOURCES memory_map_tests.cpp
  DEPENDS_ON umpire_util umpire_tpl_judy gtest
  OUTPUT_DIR ${util_dir})

blt_add_test(
  NAME memory_map_tests
  COMMAND memory_map_tests)
