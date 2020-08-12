##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (strategy_tests_depends
  umpire gtest)

if (ENABLE_HIP)
  set (strategy_tests_depends ${strategy_tests_depends} hip_runtime)
endif()

if (ENABLE_OPENMP_TARGET)
  set (strategy_tests_depends
    ${strategy_tests_depends}
    openmp)
endif ()

blt_add_executable(
  NAME fixed_pool_tests
  SOURCES fixed_pool_tests.cpp
  DEPENDS_ON ${strategy_tests_depends})

blt_add_test(
  NAME fixed_pool_tests
  COMMAND fixed_pool_tests)
