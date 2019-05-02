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
set (unit_dir ${UMPIRE_TESTS_OUTPUT_DIR}/tpl)

blt_add_executable(
  NAME judy_smoke_tests
  SOURCES judy_smoke_tests.cpp
  DEPENDS_ON umpire_tpl_judy gtest
  OUTPUT_DIR ${unit_dir})

blt_add_test(
  NAME judy_smoke_tests
  COMMAND judy_smoke_tests)

blt_add_executable(
  NAME simpool_smoke_tests
  SOURCES simpool_smoke_tests.cpp
  DEPENDS_ON umpire_tpl_simpool gtest
  OUTPUT_DIR ${unit_dir})

blt_add_test(
  NAME simpool_smoke_tests
  COMMAND simpool_smoke_tests)

blt_add_executable(
  NAME cxxopts_smoke_tests
  SOURCES cxxopts_smoke_tests.cpp
  DEPENDS_ON umpire_tpl_cxxopts gtest
  OUTPUT_DIR ${unit_dir})

blt_add_test(
  NAME cxxopts_smoke_tests
  COMMAND cxxopts_smoke_tests)
