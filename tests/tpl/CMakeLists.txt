##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
blt_add_executable(
  NAME judy_smoke_tests
  SOURCES judy_smoke_tests.cpp
  DEPENDS_ON umpire_tpl_judy gtest)

blt_add_test(
  NAME judy_smoke_tests
  COMMAND judy_smoke_tests)

blt_add_executable(
  NAME cli11_smoke_test
  SOURCES cli11_smoke_test.cpp
  DEPENDS_ON umpire_tpl_CLI11 gtest)

blt_add_test(
  NAME cli11_smoke_help
  COMMAND cli11_smoke_test)