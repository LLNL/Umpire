##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (alloc_output_dir ${TEST_OUTPUT_DIRECTORY}/unit/alloc)

set (memory_allocator_tests_depends
  umpire gtest)

if (ENABLE_CUDA)
  set (memory_allocator_tests_depends
    ${memory_allocator_tests_depends}
    cuda_runtime)
endif ()
if (ENABLE_HIP)
  set (memory_allocator_tests_depends
    ${memory_allocator_tests_depends}
    hip_runtime)
endif ()

blt_add_executable(
  NAME memory_allocator_tests
  SOURCES memory_allocator_tests.cpp
  DEPENDS_ON ${memory_allocator_tests_depends}
  OUTPUT_DIR ${alloc_output_dir})

target_include_directories(
  memory_allocator_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME memory_allocator_tests
  COMMAND memory_allocator_tests)
