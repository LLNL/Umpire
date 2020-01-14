##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
if (ENABLE_C)
  blt_add_executable(
    NAME allocator_c_tests
    SOURCES allocator_c_tests.cpp
    DEPENDS_ON umpire gtest)

  target_include_directories(
    allocator_c_tests
    PRIVATE
    ${PROJECT_BINARY_DIR}/include)

  blt_add_test(
    NAME allocator_c_tests
    COMMAND allocator_c_tests)

  blt_add_executable(
    NAME operation_c_tests
    SOURCES operation_c_tests.cpp
    DEPENDS_ON umpire gtest)

  target_include_directories(
    operation_c_tests
    PRIVATE
    ${PROJECT_BINARY_DIR}/include)

  blt_add_test(
    NAME operation_c_tests
    COMMAND operation_c_tests)

  if (ENABLE_FORTRAN)
    add_subdirectory(fortran)
  endif()
endif()
