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
set (allocator_integration_tests_depends
  umpire gtest)

if (ENABLE_CUDA)
  set (allocator_integration_tests_depends
    ${allocator_integration_tests_depends}
    cuda_runtime)
endif ()

blt_add_executable(
  NAME allocator_integration_tests
  SOURCES allocator_integration_tests.cpp
  DEPENDS_ON ${allocator_integration_tests_depends}
  OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

target_include_directories(
  allocator_integration_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME allocator_integration_tests
  COMMAND allocator_integration_tests)

set (strategy_test_depends 
  umpire gtest)

if (ENABLE_OPENMP)
  set (strategy_test_depends
    ${strategy_test_depends}
    openmp)
endif ()

blt_add_executable(
  NAME strategy_tests
  SOURCES strategy_tests.cpp
  DEPENDS_ON ${strategy_test_depends}
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

blt_add_executable(
  NAME free_functions_integration_tests
  SOURCES free_functions_integration_tests.cpp
  DEPENDS_ON umpire gtest
  OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})

target_include_directories(
  free_functions_integration_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME free_functions_integration_tests
  COMMAND free_functions_integration_tests)

add_subdirectory(replay)

# blt_add_executable(
#   NAME memory_resource_traits_tests
#   SOURCES memory_resource_traits_tests.cpp
#   DEPENDS_ON umpire gtest
#   OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR})
# 
# target_include_directories(
#   memory_resource_traits_tests
#   PRIVATE
#   ${PROJECT_BINARY_DIR}/include)
# 
# blt_add_test(
#   NAME memory_resource_traits_tests
#   COMMAND memory_resource_traits_tests)
