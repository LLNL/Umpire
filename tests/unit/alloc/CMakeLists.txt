##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (memory_allocator_tests_depends
  umpire gtest)

if (UMPIRE_ENABLE_CUDA)
  set (memory_allocator_tests_depends
    ${memory_allocator_tests_depends}
    cuda_runtime)
endif ()

blt_add_executable(
  NAME memory_allocator_tests
  SOURCES memory_allocator_tests.cpp
  DEPENDS_ON ${memory_allocator_tests_depends})

target_include_directories(
  memory_allocator_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME memory_allocator_tests
  COMMAND memory_allocator_tests)
