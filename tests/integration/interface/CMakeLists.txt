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
set (interface_dir ${UMPIRE_TESTS_OUTPUT_DIR}/integration/interface)

if (ENABLE_C)
  blt_add_executable(
    NAME allocator_c_tests
    SOURCES allocator_c_tests.cpp
    DEPENDS_ON umpire gtest
    OUTPUT_DIR ${interface_dir})

  target_include_directories(
    allocator_c_tests
    PRIVATE
    ${PROJECT_BINARY_DIR}/include)

  blt_add_test(
    NAME allocator_c_tests
    COMMAND allocator_c_tests)

  blt_add_executable(
    NAME operation_c_tests
    SOURCES operation_c_tests.cpp
    DEPENDS_ON umpire gtest
    OUTPUT_DIR ${interface_dir})

  target_include_directories(
    operation_c_tests
    PRIVATE
    ${PROJECT_BINARY_DIR}/include)

  blt_add_test(
    NAME operation_c_tests
    COMMAND operation_c_tests)

  if (ENABLE_FORTRAN)
    blt_add_executable(
      NAME fortran_interface_tests
      SOURCES umpire_fortran_interface.f
      DEPENDS_ON fruit umpire
    )

    set_source_files_properties(
      umpire_fortran_interface.f
      PROPERTIES
      Fortran_FORMAT FREE)

    blt_add_executable(
      NAME umpire_fortran_tests
      SOURCES umpire_fortran_tests.f
      DEPENDS_ON fruit umpire
    )

    set_source_files_properties(
      umpire_fortran_tests.f
      PROPERTIES
      Fortran_FORMAT FREE)
  endif()
endif()
