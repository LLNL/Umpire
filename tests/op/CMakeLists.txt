##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (opator_tests_depends
  umpire_resource umpire_op gtest)

if (ENABLE_CUDA)
  set (op_tests_depends
    ${op_tests_depends}
    cuda_runtime)
endif ()

if (ENABLE_HIP)
  set (op_tests_depends
    ${op_tests_depends}
    hip_runtime)
endif ()

blt_add_executable(
  NAME op_tests
  SOURCES op_tests.cpp
  DEPENDS_ON ${opator_tests_depends})

target_include_directories(
  op_tests
  PRIVATE
  ${PROJECT_BINARY_DIR}/include)

blt_add_test(
  NAME op_tests
  COMMAND op_tests)