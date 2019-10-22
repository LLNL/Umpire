##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
if (ENABLE_CUDA)
  blt_add_executable(
    NAME cuda_mem_advise_op_tests
    SOURCES cuda_mem_advise_op_tests.cpp
    DEPENDS_ON umpire cuda_runtime gtest gmock)

  blt_add_test(
    NAME cuda_mem_advise_op_tests
    COMMAND cuda_mem_advise_op_tests)
endif ()
