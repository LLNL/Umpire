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
set (resource_dir ${UMPIRE_TESTS_OUTPUT_DIR}/unit/resource)

set (resource_depends umpire gtest gmock)

if (ENABLE_HIP)
  set (resource_depends ${resource_depends} hip_runtime)
endif()

if (ENABLE_GMOCK)
  blt_add_executable(
    NAME resource_tests
    SOURCES resource_tests.cpp
    DEPENDS_ON ${resource_depends}
    OUTPUT_DIR ${resource_dir})

  blt_add_test(
    NAME resource_tests
    COMMAND resource_tests)
endif ()
