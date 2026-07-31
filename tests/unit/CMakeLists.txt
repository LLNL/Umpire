##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (unit_depends umpire gtest)

if (UMPIRE_ENABLE_OPENMP_TARGET)
  set (unit_depends ${unit_depends} openmp)
endif ()

blt_add_executable(
  NAME resource_manager_tests
  SOURCES resource_manager_tests.cpp
  DEPENDS_ON ${unit_depends})

blt_add_test(
  NAME resource_manager_tests
  COMMAND resource_manager_tests)

blt_add_executable(
  NAME version_tests
  SOURCES version_tests.cpp
  DEPENDS_ON ${unit_depends})

blt_add_test(
  NAME version_tests
  COMMAND version_tests)

blt_add_executable(
  NAME umpire_tests
  SOURCES umpire_tests.cpp
  DEPENDS_ON ${unit_depends})

blt_add_test(
  NAME umpire_tests
  COMMAND umpire_tests)

if (UMPIRE_ENABLE_API_V2)
  blt_add_executable(
    NAME api_v2_registry_tests
    SOURCES api_v2/test_registry.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_registry_tests
    COMMAND api_v2_registry_tests)

  blt_add_executable(
    NAME api_v2_memory_tests
    SOURCES api_v2/test_memory.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_memory_tests
    COMMAND api_v2_memory_tests)

  blt_add_executable(
    NAME api_v2_memory_resource_tests
    SOURCES api_v2/test_memory_resource.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_memory_resource_tests
    COMMAND api_v2_memory_resource_tests)

  blt_add_executable(
    NAME api_v2_allocator_tests
    SOURCES api_v2/test_allocator.cpp
    DEPENDS_ON ${unit_depends})

  target_include_directories(
    api_v2_allocator_tests
    BEFORE PRIVATE
    ${PROJECT_SOURCE_DIR}/include)

  blt_add_test(
    NAME api_v2_allocator_tests
    COMMAND api_v2_allocator_tests)

  blt_add_executable(
    NAME api_v2_allocation_strategy_tests
    SOURCES api_v2/test_allocation_strategy.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_allocation_strategy_tests
    COMMAND api_v2_allocation_strategy_tests)

  blt_add_executable(
    NAME api_v2_host_memory_tests
    SOURCES api_v2/resources/test_host_memory.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_host_memory_tests
    COMMAND api_v2_host_memory_tests)

  blt_add_executable(
    NAME api_v2_null_resource_tests
    SOURCES api_v2/resources/test_null_resource.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_null_resource_tests
    COMMAND api_v2_null_resource_tests)

  if (UMPIRE_ENABLE_FILE_RESOURCE)
    blt_add_executable(
      NAME api_v2_file_memory_tests
      SOURCES api_v2/resources/test_file_memory.cpp
      DEPENDS_ON ${unit_depends})

    blt_add_test(
      NAME api_v2_file_memory_tests
      COMMAND api_v2_file_memory_tests)
  endif()

  if (UMPIRE_ENABLE_SHARED_MEMORY)
    blt_add_executable(
      NAME api_v2_shared_memory_tests
      SOURCES api_v2/resources/test_shared_memory.cpp
      DEPENDS_ON ${unit_depends})

    blt_add_test(
      NAME api_v2_shared_memory_tests
      COMMAND api_v2_shared_memory_tests)
  endif()

  if (UMPIRE_ENABLE_CUDA)
    blt_add_executable(
      NAME api_v2_cuda_device_memory_tests
      SOURCES api_v2/resources/test_cuda_device_memory.cpp
      DEPENDS_ON ${unit_depends} cuda_runtime)

    blt_add_test(
      NAME api_v2_cuda_device_memory_tests
      COMMAND api_v2_cuda_device_memory_tests)
  endif()

  if (UMPIRE_ENABLE_HIP)
    blt_add_executable(
      NAME api_v2_hip_device_memory_tests
      SOURCES api_v2/resources/test_hip_device_memory.cpp
      DEPENDS_ON ${unit_depends} hip)

    blt_add_test(
      NAME api_v2_hip_device_memory_tests
      COMMAND api_v2_hip_device_memory_tests)
  endif()

  if (UMPIRE_ENABLE_SYCL)
    blt_add_executable(
      NAME api_v2_sycl_device_memory_tests
      SOURCES api_v2/resources/test_sycl_device_memory.cpp
      DEPENDS_ON ${unit_depends})

    blt_add_test(
      NAME api_v2_sycl_device_memory_tests
      COMMAND api_v2_sycl_device_memory_tests)
  endif()

  if (UMPIRE_ENABLE_CUDA OR UMPIRE_ENABLE_HIP OR UMPIRE_ENABLE_SYCL)
    set (api_v2_um_memory_depends ${unit_depends})
    if (UMPIRE_ENABLE_CUDA)
      list(APPEND api_v2_um_memory_depends cuda_runtime)
    endif()
    if (UMPIRE_ENABLE_HIP)
      list(APPEND api_v2_um_memory_depends hip)
    endif()

    blt_add_executable(
      NAME api_v2_um_memory_tests
      SOURCES api_v2/resources/test_um_memory.cpp
      DEPENDS_ON ${api_v2_um_memory_depends})

    blt_add_test(
      NAME api_v2_um_memory_tests
      COMMAND api_v2_um_memory_tests)

    set (api_v2_pinned_memory_depends ${unit_depends})
    if (UMPIRE_ENABLE_CUDA)
      list(APPEND api_v2_pinned_memory_depends cuda_runtime)
    endif()
    if (UMPIRE_ENABLE_HIP)
      list(APPEND api_v2_pinned_memory_depends hip)
    endif()

    blt_add_executable(
      NAME api_v2_pinned_memory_tests
      SOURCES api_v2/resources/test_pinned_memory.cpp
      DEPENDS_ON ${api_v2_pinned_memory_depends})

    blt_add_test(
      NAME api_v2_pinned_memory_tests
      COMMAND api_v2_pinned_memory_tests)
  endif()

  if (UMPIRE_ENABLE_CUDA OR UMPIRE_ENABLE_HIP)
    set (api_v2_device_const_memory_depends ${unit_depends})
    if (UMPIRE_ENABLE_CUDA)
      list(APPEND api_v2_device_const_memory_depends cuda_runtime)
    endif()
    if (UMPIRE_ENABLE_HIP)
      list(APPEND api_v2_device_const_memory_depends hip)
    endif()

    blt_add_executable(
      NAME api_v2_device_const_memory_tests
      SOURCES api_v2/resources/test_device_const_memory.cpp
      DEPENDS_ON ${api_v2_device_const_memory_depends})

    blt_add_test(
      NAME api_v2_device_const_memory_tests
      COMMAND api_v2_device_const_memory_tests)
  endif()

  if (UMPIRE_ENABLE_OPENMP_TARGET)
    blt_add_executable(
      NAME api_v2_openmp_target_memory_tests
      SOURCES api_v2/resources/test_openmp_target_memory.cpp
      DEPENDS_ON ${unit_depends})

    blt_add_test(
      NAME api_v2_openmp_target_memory_tests
      COMMAND api_v2_openmp_target_memory_tests)
  endif()

  blt_add_executable(
    NAME api_v2_thread_safe_tests
    SOURCES api_v2/strategies/test_thread_safe.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_thread_safe_tests
    COMMAND api_v2_thread_safe_tests)

  blt_add_executable(
    NAME api_v2_fixed_pool_tests
    SOURCES api_v2/strategies/test_fixed_pool.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_fixed_pool_tests
    COMMAND api_v2_fixed_pool_tests)

  blt_add_executable(
    NAME api_v2_coalescing_pool_list_tests
    SOURCES api_v2/strategies/test_coalescing_pool_list.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_coalescing_pool_list_tests
    COMMAND api_v2_coalescing_pool_list_tests)

  blt_add_executable(
    NAME api_v2_binned_pool_tests
    SOURCES api_v2/strategies/test_binned_pool.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_binned_pool_tests
    COMMAND api_v2_binned_pool_tests)

  blt_add_executable(
    NAME api_v2_quick_pool_tests
    SOURCES api_v2/strategies/test_quick_pool.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_quick_pool_tests
    COMMAND api_v2_quick_pool_tests)

  blt_add_executable(
    NAME api_v2_dynamic_pool_list_tests
    SOURCES api_v2/strategies/test_dynamic_pool_list.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_dynamic_pool_list_tests
    COMMAND api_v2_dynamic_pool_list_tests)

  blt_add_executable(
    NAME api_v2_size_limiter_tests
    SOURCES api_v2/strategies/test_size_limiter.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_size_limiter_tests
    COMMAND api_v2_size_limiter_tests)

  blt_add_executable(
    NAME api_v2_monotonic_buffer_tests
    SOURCES api_v2/strategies/test_monotonic_buffer.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_monotonic_buffer_tests
    COMMAND api_v2_monotonic_buffer_tests)

  blt_add_executable(
    NAME api_v2_named_tests
    SOURCES api_v2/strategies/test_named.cpp
    DEPENDS_ON ${unit_depends})

  blt_add_test(
    NAME api_v2_named_tests
    COMMAND api_v2_named_tests)
endif ()

add_subdirectory(alloc)
add_subdirectory(op)
add_subdirectory(resource)
add_subdirectory(util)
add_subdirectory(strategy)
