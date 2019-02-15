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
if (ENABLE_GMOCK)
  blt_add_executable(
    NAME resource_tests
    SOURCES resource_tests.cpp
    DEPENDS_ON umpire umpire_resource umpire_util umpire gtest gmock
    OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR}/unit/resource)

  blt_add_test(
    NAME resource_tests
    COMMAND resource_tests)
endif ()
