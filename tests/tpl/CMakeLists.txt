##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
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
  NAME cxxopts_smoke_tests
  SOURCES cxxopts_smoke_tests.cpp
  DEPENDS_ON umpire_tpl_cxxopts gtest
  OUTPUT_DIR ${unit_dir})

if (NOT C_COMPILER_FAMILY_IS_PGI)
  blt_add_test(
    NAME cxxopts_smoke_tests
    COMMAND cxxopts_smoke_tests)
endif()
