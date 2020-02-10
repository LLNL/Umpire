##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (fortran_test_defines)

if (ENABLE_CUDA)
  set (fortran_test_defines -DUMPIRE_ENABLE_CUDA)
endif ()

blt_add_executable(
  NAME allocator_fortran_tests
  SOURCES allocator_fortran_tests.F
  DEFINES ${fortran_test_defines}
  DEPENDS_ON fruit umpire)

set_source_files_properties(
  allocator_fortran_tests.F
  PROPERTIES
  Fortran_FORMAT FREE)

blt_add_test(
  NAME allocator_fortran_tests
  COMMAND allocator_fortran_tests)

blt_add_executable(
  NAME op_fortran_tests
  SOURCES op_fortran_tests.F
  DEFINES ${fortran_test_defines}
  DEPENDS_ON fruit umpire)

set_source_files_properties(
  op_fortran_tests.F
  PROPERTIES
  Fortran_FORMAT FREE)

blt_add_test(
  NAME op_fortran_tests
  COMMAND op_fortran_tests)
