##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (resource_tests_depends
  umpire umpire_resource umpire_detail gtest)

if (ENABLE_CUDA)
  set (resource_tests_depends
    ${resource_tests_depends}
    cuda_runtime)
endif ()

if (ENABLE_HIP)
  set (resource_tests_depends
    ${resource_tests_depends}
    hip_runtime)
endif ()

blt_add_executable(
  NAME resource_tests
  SOURCES resource_tests.cpp
  DEPENDS_ON ${resource_tests_depends})

target_include_directories(
  resource_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME resource_tests
  COMMAND resource_tests)
