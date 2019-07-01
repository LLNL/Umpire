##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (strategy_dir ${UMPIRE_TESTS_OUTPUT_DIR}/unit/strategy)

set (strategy_tests_depends
  umpire gtest)

if (ENABLE_HIP)
  set (strategy_tests_depends ${strategy_tests_depends} hip_runtime)
endif()

blt_add_executable(
  NAME fixed_pool_tests
  SOURCES fixed_pool_tests.cpp
  DEPENDS_ON ${strategy_tests_depends}
  OUTPUT_DIR ${strategy_dir})

target_include_directories(
  memory_allocator_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME fixed_pool_tests
  COMMAND fixed_pool_tests)
