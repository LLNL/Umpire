##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
if (UMPIRE_ENABLE_CUDA OR UMPIRE_ENABLE_HIP)
  blt_add_executable(
    NAME mem_advise_op_tests
    SOURCES mem_advise_op_tests.cpp
    DEPENDS_ON umpire gtest gmock)

  blt_add_test(
    NAME mem_advise_op_tests
    COMMAND mem_advise_op_tests)
endif ()
