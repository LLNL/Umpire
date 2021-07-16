##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (allocator_tests_depends
  umpire_resource umpire_alloc umpire_tracking umpire_util umpire_tpl_judy gtest)

if (ENABLE_CUDA)
  set (allocator_tests_depends
    ${allocator_tests_depends}
    cuda_runtime)
endif ()

if (ENABLE_HIP)
  set (allocator_tests_depends
    ${allocator_tests_depends}
    hip_runtime)
endif ()

blt_add_executable(
  NAME allocator_tests
  SOURCES allocator_tests.cpp
  DEPENDS_ON ${allocator_tests_depends})

target_include_directories(
  allocator_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME allocator_tests
  COMMAND allocator_tests)