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
set (util_dir ${UMPIRE_TEST_OUTPUT_DIR}/unit/util)

blt_add_executable(
  NAME exception_tests
  SOURCES exception_tests.cpp
  DEPENDS_ON umpire_util gtest
  OUTPUT_DIR ${util_dir})

blt_add_test(
  NAME exception_tests
  COMMAND exception_tests)

blt_add_executable(
  NAME allocation_map_tests
  SOURCES allocation_map_tests.cpp
  DEPENDS_ON umpire_util gtest
  OUTPUT_DIR ${util_dir})

blt_add_test(
  NAME allocation_map_tests
  COMMAND allocation_map_tests)

# TODO: Make this test a relative performance test
if (False)
  blt_add_executable(
    NAME allocation_map_performance_tests
    SOURCES allocation_map_performance_tests.cpp
    DEPENDS_ON umpire_util gtest
    OUTPUT_DIR ${util_dir})

  blt_add_test(
    NAME allocation_map_performance_tests
    COMMAND allocation_map_performance_tests)
endif ()
