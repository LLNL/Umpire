##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (resource_depends umpire gtest gmock)

blt_add_executable(
  NAME null_resource_tests
  SOURCES null_resource_tests.cpp
  DEPENDS_ON umpire gtest)

blt_add_test(
  NAME null_resource_tests
  COMMAND null_resource_tests)

if(UMPIRE_ENABLE_FILE_RESOURCE)
  blt_add_executable(
    NAME file_resource_tests
    SOURCES file_resource_tests.cpp
    DEPENDS_ON umpire gtest)

  blt_add_test(
    NAME file_resource_tests
    COMMAND file_resource_tests)
endif()

if(UMPIRE_ENABLE_IPC_SHARED_MEMORY AND UMPIRE_ENABLE_MPI)
  blt_add_executable(
    NAME ipc_shared_memory_resource_tests
    SOURCES ipc_shared_memory_resource_tests.cpp
    DEPENDS_ON ${resource_depends})

  umpire_add_test_with_mpi(
    NAME ipc_shared_memory_resource_tests
    COMMAND ipc_shared_memory_resource_tests
    NUM_MPI_TASKS 4)
endif()

if (UMPIRE_ENABLE_MPI3_SHARED_MEMORY)
  blt_add_executable(
    NAME mpi3_shared_memory_resource_tests
    SOURCES mpi3_shared_memory_resource_tests.cpp
    DEPENDS_ON ${resource_depends})

  umpire_add_test_with_mpi(
    NAME mpi3_shared_memory_resource_tests
    COMMAND mpi3_shared_memory_resource_tests
    NUM_MPI_TASKS 4)
endif ()

if (UMPIRE_ENABLE_GMOCK)
  blt_add_executable(
    NAME resource_tests
    SOURCES resource_tests.cpp
    DEPENDS_ON ${resource_depends})

  blt_add_test(
    NAME resource_tests
    COMMAND resource_tests)
endif ()
