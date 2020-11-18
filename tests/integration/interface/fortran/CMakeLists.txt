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

if (ENABLE_HIP)
  set (fortran_test_defines -DUMPIRE_ENABLE_HIP)
endif ()

set (fortran_tests_depends
  umpire fruit)

if (ENABLE_CUDA)
  set (fortran_tests_depends
    ${fortran_tests_depends}
    cuda_runtime)
endif ()

if (ENABLE_HIP)
  set (fortran_tests_depends
    ${fortran_tests_depends}
    hip_runtime)
endif ()

if (ENABLE_OPENMP OR ENABLE_OPENMP_TARGET)
  set (fortran_tests_depends
    ${fortran_tests_depends}
    openmp)
endif ()

blt_add_executable(
  NAME allocator_fortran_tests
  SOURCES allocator_fortran_tests.F
  DEFINES ${fortran_test_defines}
  DEPENDS_ON ${fortran_tests_depends})

set_source_files_properties(
  allocator_fortran_tests.F
  PROPERTIES
  Fortran_FORMAT FREE)

blt_add_test(
  NAME allocator_fortran_tests
  COMMAND allocator_fortran_tests)

#Until more stable blt changes can be made, this is a temporary fix to allow
#the fortran program to be linked with hip (https://rzlc.llnl.gov/jira/browse/UM-790)
if(ENABLE_HIP)
  set_target_properties(allocator_fortran_tests PROPERTIES LINKER_LANGUAGE HIP)
endif()

blt_add_executable(
  NAME op_fortran_tests
  SOURCES op_fortran_tests.F
  DEFINES ${fortran_test_defines}
  DEPENDS_ON ${fortran_tests_depends})

set_source_files_properties(
  op_fortran_tests.F
  PROPERTIES
  Fortran_FORMAT FREE)

blt_add_test(
  NAME op_fortran_tests
  COMMAND op_fortran_tests)

if(ENABLE_HIP)
  set_target_properties(op_fortran_tests PROPERTIES LINKER_LANGUAGE HIP)
endif()

blt_add_executable(
  NAME introspection_fortran_tests
  SOURCES introspection_fortran_tests.F
  DEFINES ${fortran_test_defines}
  DEPENDS_ON ${fortran_tests_depends})

set_source_files_properties(
  introspection_fortran_tests.F
  PROPERTIES
  Fortran_FORMAT FREE)

blt_add_test(
  NAME introspection_fortran_tests
  COMMAND introspection_fortran_tests)

if(ENABLE_HIP)
  set_target_properties(introspection_fortran_tests PROPERTIES LINKER_LANGUAGE HIP)
endif()
