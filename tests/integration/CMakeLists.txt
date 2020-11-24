##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (integration_tests_depends
  umpire gtest)
set (accessibility_test_depends
  umpire gtest)

if (ENABLE_CUDA)
  set (integration_tests_depends
    ${integration_tests_depends}
    cuda_runtime)
endif ()

if (ENABLE_HIP)
  set (integration_tests_depends
    ${integration_tests_depends}
    hip_runtime)
endif ()

if (ENABLE_OPENMP OR ENABLE_OPENMP_TARGET)
  set (integration_tests_depends
    ${integration_tests_depends}
    openmp)
endif ()

if (ENABLE_CUDA)
  set (accessibility_test_depends
    ${accessibility_test_depends}
    cuda)
endif ()

if (ENABLE_HIP)
  set (accessibility_test_depends
    ${accessibility_test_depends}
    hip)
endif ()

if (ENABLE_OPENMP OR ENABLE_OPENMP_TARGET)
  set (accessibility_test_depends
    ${accessibility_test_depends}
    openmp)
endif ()

blt_add_executable(
  NAME allocator_integration_tests
  SOURCES allocator_integration_tests.cpp
  DEPENDS_ON ${integration_tests_depends})

target_include_directories(
  allocator_integration_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME allocator_integration_tests
  COMMAND allocator_integration_tests)

blt_add_executable(
  NAME allocator_accessibility_tests
  SOURCES allocator_accessibility.cpp
  DEPENDS_ON ${accessibility_test_depends})

target_include_directories(
  allocator_accessibility_tests
  PRIVATE 
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME allocator_accessibility_tests
  COMMAND allocator_accessibility_tests)

blt_add_executable(
  NAME memory_resource_traits_tests 
  SOURCES memory_resource_traits_tests.cpp
  DEPENDS_ON ${integration_tests_depends})

target_include_directories(
  memory_resource_traits_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME memory_resource_traits_tests
  COMMAND memory_resource_traits_tests)

blt_add_executable(
  NAME strategy_tests
  SOURCES strategy_tests.cpp
  DEPENDS_ON ${integration_tests_depends})

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
  DEPENDS_ON ${integration_tests_depends})

target_include_directories(
  operation_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME operation_tests
  COMMAND operation_tests)

blt_add_executable(
  NAME reallocate_tests
  SOURCES reallocate_tests.cpp
  DEPENDS_ON ${integration_tests_depends})

target_include_directories(
  reallocate_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME reallocate_tests
  COMMAND reallocate_tests)

blt_add_executable(
  NAME free_functions_integration_tests
  SOURCES free_functions_integration_tests.cpp
  DEPENDS_ON ${integration_tests_depends})

target_include_directories(
  free_functions_integration_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME free_functions_integration_tests
  COMMAND free_functions_integration_tests)

blt_add_executable(
  NAME primary_pool_tests
  SOURCES primary_pool_tests.cpp
  DEPENDS_ON ${integration_tests_depends})

blt_add_test(
  NAME primary_pool_tests
  COMMAND primary_pool_tests)

if (ENABLE_PERFORMANCE_TESTS)
  blt_add_executable(
    NAME primary_pool_performance_tests
    SOURCES primary_pool_performance_tests.cpp
    DEPENDS_ON ${integration_tests_depends})

  blt_add_test(
    NAME primary_pool_performance_tests
    COMMAND primary_pool_performance_tests)
endif()

blt_add_executable(
  NAME static_initialization_test
  SOURCES static_initialization_test.cpp
  DEPENDS_ON ${integration_tests_depends})

blt_add_test(
  NAME static_initialization_test
  COMMAND static_initialization_test)

blt_add_executable(
  NAME introspection_tests
  SOURCES introspection_tests.cpp
  DEPENDS_ON ${integration_tests_depends})

blt_add_test(
  NAME introspection_tests
  COMMAND introspection_tests)

if (ENABLE_TOOLS)
  add_subdirectory(replay)
endif()

if (ENABLE_C OR ENABLE_FORTRAN)
  add_subdirectory(interface)
endif()

add_subdirectory(io)
