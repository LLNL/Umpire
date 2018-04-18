##############################################################################
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
set (allocator_tests_depends
  umpire gtest)

if (ENABLE_CUDA)
  set (allocator_tests_depends
    ${allocator_tests_depends}
    cuda_runtime)
endif ()

blt_add_executable(
  NAME allocator_tests
  SOURCES allocator_tests.cpp
  DEPENDS_ON ${allocator_tests_depends}
  OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

target_include_directories(
  allocator_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME allocator_tests
  COMMAND allocator_tests)

if (ENABLE_C)
  blt_add_executable(
    NAME allocator_c_tests
    SOURCES allocator_c_tests.cpp
    DEPENDS_ON ${allocator_tests_depends}
    OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

  target_include_directories(
    allocator_c_tests
    PRIVATE
    ${PROJECT_BINARY_DIR}/include)

  blt_add_test(
    NAME allocator_c_tests
    COMMAND allocator_c_tests)

endif()

blt_add_executable(
  NAME strategy_tests
  SOURCES strategy_tests.cpp
  DEPENDS_ON umpire gtest
  OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

target_include_directories(
  strategy_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME strategy_tests
  COMMAND strategy_tests)

blt_add_executable(
  NAME operation_tests
  SOURCES operation_tests.cpp
  DEPENDS_ON umpire gtest
  OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

target_include_directories(
  operation_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME operation_tests
  COMMAND operation_tests)

if (ENABLE_ASSERTS)
  blt_add_executable(
    NAME assert_tests
    SOURCES assert_tests.cpp
    DEPENDS_ON umpire gtest
    OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

  target_include_directories(
    assert_tests
    PRIVATE
    ${PROJECT_BINARY_DIR}/include)

  blt_add_test(
    NAME assert_tests
    COMMAND assert_tests)
endif ()
