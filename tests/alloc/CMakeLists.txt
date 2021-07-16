##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (alloc_tests_depends
  umpire umpire_alloc gtest)

if (ENABLE_CUDA)
  set (alloc_tests_depends
    ${alloc_tests_depends}
    cuda_runtime)
endif ()
if (ENABLE_HIP)
  set (alloc_tests_depends
    ${alloc_tests_depends}
    hip_runtime)
endif ()

blt_add_executable(
  NAME alloc_tests
  SOURCES alloc_tests.cpp
  DEPENDS_ON ${alloc_tests_depends})

target_include_directories(
  alloc_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME alloc_tests
  COMMAND alloc_tests)
