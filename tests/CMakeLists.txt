##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

# With intel oneapi (as of version 2022.[1|2]) we see errors building gtest
# due to the default flag "-ffp-model=fast" (relates to comparing to NaN).
if(C_COMPILER_FAMILY_IS_INTELLLVM)
  target_compile_options(gtest PRIVATE -ffp-model=precise)
  target_compile_options(gtest_main PRIVATE -ffp-model=precise)
endif()

add_subdirectory(tpl)
add_subdirectory(unit)
add_subdirectory(integration)
add_subdirectory(tools)
add_subdirectory(debug)
add_subdirectory(op)
if (UMPIRE_ENABLE_DEVELOPER_DEFAULTS)
    add_subdirectory(applications)
endif()
add_subdirectory(install)

if (UMPIRE_ENABLE_API_V2
    AND (ENABLE_COVERAGE OR UMPIRE_ENABLE_COVERAGE)
    AND EXISTS "${LCOV_EXECUTABLE}"
    AND EXISTS "${GENHTML_EXECUTABLE}"
    AND EXISTS "${GCOV_EXECUTABLE}")
  set(api_v2_coverage_targets
    api_v2_registry_tests
    api_v2_memory_tests
    api_v2_memory_resource_tests
    api_v2_allocator_tests
    api_v2_allocation_strategy_tests
    api_v2_host_memory_tests
    api_v2_null_resource_tests
    api_v2_fixed_pool_tests
    api_v2_thread_safe_tests
    api_v2_dynamic_pool_list_tests
    api_v2_quick_pool_tests
    api_v2_size_limiter_tests
    api_v2_monotonic_buffer_tests
    api_v2_named_tests
    api_v2_registry_threading_tests
    api_v2_memory_threading_tests
    api_v2_thread_safe_stress_tests
    api_v2_stl_compatibility_tests
    api_v2_operations_headers_tests
    api_v2_operations_tests
    api_v2_v1_interop_tests)
  set(API_V2_COVERAGE_EXCLUDED_TESTS "")
  configure_file(
    ${PROJECT_SOURCE_DIR}/cmake/RunApiV2CoverageTests.cmake.in
    ${PROJECT_BINARY_DIR}/RunApiV2CoverageTests.cmake
    @ONLY)

  add_custom_target(coverage_api_v2
    COMMAND
      ${LCOV_EXECUTABLE}
      --gcov-tool ${GCOV_EXECUTABLE}
      --directory=${PROJECT_BINARY_DIR}
      --zerocounters
    COMMAND
      ${CMAKE_COMMAND}
      -P ${PROJECT_BINARY_DIR}/RunApiV2CoverageTests.cmake
    COMMAND
      ${LCOV_EXECUTABLE}
      --ignore-errors empty,inconsistent,unsupported,format
      --no-external
      --gcov-tool ${GCOV_EXECUTABLE}
      --base-directory ${PROJECT_SOURCE_DIR}
      --directory=${PROJECT_BINARY_DIR}
      --capture
      --output-file coverage_api_v2.info
    COMMAND
      ${CMAKE_COMMAND}
      -E copy
      coverage_api_v2.info
      coverage_api_v2.info.cleaned
    COMMAND
      ${GENHTML_EXECUTABLE}
      --ignore-errors inconsistent,corrupt,category
      -o coverage_api_v2
      coverage_api_v2.info.cleaned
    BYPRODUCTS
      coverage_api_v2.info
      coverage_api_v2.info.cleaned
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    COMMENT "Resetting API v2 coverage counters, running host-safe API v2 tests, and generating coverage report.")
  add_dependencies(coverage_api_v2 ${api_v2_coverage_targets})

  message(STATUS "Coverage target 'coverage_api_v2' available for API v2 tests.")
endif()

umpire_add_code_checks(PREFIX tests RECURSIVE TRUE)
