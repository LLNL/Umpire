##############################################################################
# Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
#
# Created by David Beckingsale, david@llnl.gov
# LLNL-CODE-747640
#
# All rights reserved.
#
# This file is part of Umpire.
#
# For details, see https://github.com/LLNL/Umpire
# Please also see the LICENSE file for MIT license.
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
