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
set (alloc_dir ${UMPIRE_TEST_OUTPUT_DIR}/unit/alloc)

set (memory_allocator_tests_depends
  umpire_alloc gtest)

if (ENABLE_CUDA)
  set (memory_allocator_tests_depends
    ${memory_allocator_tests_depends}
    cuda_runtime)
endif ()

blt_add_executable(
  NAME memory_allocator_tests
  SOURCES memory_allocator_tests.cpp
  DEPENDS_ON ${memory_allocator_tests_depends}
  OUTPUT_DIR ${alloc_dir})

target_include_directories(
  memory_allocator_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME memory_allocator_tests
  COMMAND memory_allocator_tests)
