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
set (fortran_test_defines)

if (ENABLE_CUDA)
  set (fortran_test_defines -DUMPIRE_ENABLE_CUDA)
endif ()

blt_add_executable(
  NAME allocator_fortran_tests
  SOURCES allocator_fortran_tests.F
  DEFINES ${fortran_test_defines}
  DEPENDS_ON fruit umpire
  OUTPUT_DIR ${interface_dir}
)

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
  DEPENDS_ON fruit umpire
  OUTPUT_DIR ${interface_dir}
)

set_source_files_properties(
  op_fortran_tests.F
  PROPERTIES
  Fortran_FORMAT FREE)

blt_add_test(
  NAME op_fortran_tests
  COMMAND op_fortran_tests)
