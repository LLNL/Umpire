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
if (ENABLE_CUDA)
  blt_add_executable(
    NAME cuda_mem_advise_op_tests
    SOURCES cuda_mem_advise_op_tests.cpp
    DEPENDS_ON umpire_op umpire cuda_runtime gtest gmock)

  blt_add_test(
    NAME cuda_mem_advise_op_tests
    COMMAND cuda_mem_advise_op_tests)
endif ()
