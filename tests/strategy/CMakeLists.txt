##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (strategy_tests_depends
  umpire umpire_resource umpire_alloc umpire_detail umpire_tpl_judy gtest)

if (ENABLE_CUDA)
  set (strategy_tests_depends
    ${strategy_tests_depends}
    cuda_runtime)
endif ()

if (ENABLE_HIP)
  set (strategy_tests_depends
    ${strategy_tests_depends}
    hip_runtime)
endif ()

blt_add_executable(
  NAME strategy_tests
  SOURCES strategy_tests.cpp
  DEPENDS_ON ${strategy_tests_depends})

target_include_directories(
  strategy_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME strategy_tests
  COMMAND strategy_tests)