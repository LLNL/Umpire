##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (resource_depends umpire gtest gmock)

if (ENABLE_HIP)
  set (resource_depends ${resource_depends} hip_runtime)
endif()

blt_add_executable(
  NAME null_resource_tests
  SOURCES null_resource_tests.cpp
  DEPENDS_ON umpire gtest)

blt_add_test(
  NAME null_resource_tests
  COMMAND null_resource_tests)

if(ENABLE_FILE_RESOURCE)
  blt_add_executable(
    NAME file_resource_tests
    SOURCES file_resource_tests.cpp
    DEPENDS_ON umpire gtest)

  blt_add_test(
    NAME file_resource_tests
    COMMAND file_resource_tests)
endif()

if(ENABLE_HOST_SHARED_MEMORY)
  blt_add_executable(
    NAME shared_memory_resource_tests
    SOURCES shared_memory_resource_tests.cpp
    DEPENDS_ON ${resource_depends})

  umpire_add_test_with_mpi(
    NAME shared_memory_resource_tests
    COMMAND shared_memory_resource_tests
    NUM_MPI_TASKS 4)
endif()

if (ENABLE_GMOCK)
  blt_add_executable(
    NAME resource_tests
    SOURCES resource_tests.cpp
    DEPENDS_ON ${resource_depends})

  blt_add_test(
    NAME resource_tests
    COMMAND resource_tests)
endif ()
