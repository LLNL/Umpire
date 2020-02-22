##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (allocator_integration_tests_depends
  umpire gtest)

if (ENABLE_CUDA)
  set (allocator_integration_tests_depends
    ${allocator_integration_tests_depends}
    cuda_runtime)
endif ()
if (ENABLE_HIP)
  set (allocator_integration_tests_depends
    ${allocator_integration_tests_depends}
    hip_runtime)
endif ()

blt_add_executable(
  NAME allocator_integration_tests
  SOURCES allocator_integration_tests.cpp
  DEPENDS_ON ${allocator_integration_tests_depends})

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

if (ENABLE_HIP)
  set (strategy_test_depends
      ${strategy_test_depends}
      hip_runtime)
endif()

blt_add_executable(
  NAME strategy_tests
  SOURCES strategy_tests.cpp
  DEPENDS_ON ${strategy_test_depends})

target_include_directories(
  strategy_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME strategy_tests
  COMMAND strategy_tests)

set (operation_test_depends
  umpire gtest)

if (ENABLE_HIP)
  set (operation_test_depends ${operation_test_depends} hip)
endif()

blt_add_executable(
  NAME operation_tests
  SOURCES operation_tests.cpp
  DEPENDS_ON ${operation_test_depends})

target_include_directories(
  operation_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME operation_tests
  COMMAND operation_tests)

set (free_functions_integration_tests_depends
  umpire gtest)

if (ENABLE_HIP)
  set (free_functions_integration_tests_depends
      ${free_functions_integration_tests_depends}
      hip)
endif()

blt_add_executable(
  NAME free_functions_integration_tests
  SOURCES free_functions_integration_tests.cpp
  DEPENDS_ON ${free_functions_integration_tests_depends})

target_include_directories(
  free_functions_integration_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME free_functions_integration_tests
  COMMAND free_functions_integration_tests)

set(static_initialization_test_depends umpire)

if (ENABLE_HIP)
  set (static_initialization_test_depends
      ${static_initialization_test_depends}
      hip_runtime)
endif()

blt_add_executable(
  NAME static_initialization_test
  SOURCES static_initialization_test.cpp
  DEPENDS_ON ${static_initialization_test_depends})

blt_add_test(
  NAME static_initialization_test
  COMMAND static_initialization_test)

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

if (ENABLE_BACKTRACE)
  add_subdirectory(backtrace)
endif()

if (ENABLE_TOOLS)
  add_subdirectory(replay)
endif()

if (ENABLE_C OR ENABLE_FORTRAN)
  add_subdirectory(interface)
endif()

add_subdirectory(io)
