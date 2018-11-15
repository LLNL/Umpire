##############################################################################
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
blt_add_executable(
  NAME resource_manager_tests
  SOURCES resource_manager_tests.cpp
  DEPENDS_ON umpire gtest)

blt_add_test(
  NAME resource_manager_tests
  COMMAND resource_manager_tests)

if (ENABLE_GMOCK)
  blt_add_executable(
    NAME allocator_tests
    SOURCES allocator_tests.cpp
    OUTPUT_DIR ${UMPIRE_TEST_OUTPUT_DIR}
    DEPENDS_ON umpire gtest gmock)

  blt_add_test(
    NAME allocator_tests
    COMMAND allocator_tests)
endif ()


add_subdirectory(alloc)
add_subdirectory(op)
add_subdirectory(resource)
add_subdirectory(util)
