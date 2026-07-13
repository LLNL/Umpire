##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
blt_add_executable(
  NAME io_tests
  SOURCES io_tests.cpp
  DEPENDS_ON umpire)

if (NOT C_COMPILER_FAMILY_IS_PGI)
  find_package(Python3)

  if (Python3_FOUND)
    umpire_add_test_with_mpi(
      NAME io_tests_with_basename_logging
      NUM_MPI_TASKS 1
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/io_tests_runner.py "Basename +logging" WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    umpire_add_test_with_mpi(
      NAME io_tests_with_basename_directory
      NUM_MPI_TASKS 1
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/io_tests_runner.py "Basename +Directory -logging" WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    umpire_add_test_with_mpi(
      NAME io_tests_with_basename_directory_logging
      NUM_MPI_TASKS 1
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/io_tests_runner.py "Basename +Directory +logging" WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    umpire_add_test_with_mpi(
      NAME io_tests_with_basename_directory_replay
      NUM_MPI_TASKS 1
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/io_tests_runner.py "Basename +Directory -logging" WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    umpire_add_test_with_mpi(
      NAME io_tests_with_basename_directory_logging_replay
      NUM_MPI_TASKS 1
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/io_tests_runner.py "Basename +Directory +logging" WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endif ()
endif ()

set (log_depends umpire)

if (UMPIRE_ENABLE_OPENMP_TARGET)
  set (log_depends
    ${log_depends}
    openmp)
endif ()

blt_add_executable(
  NAME log_tests
  SOURCES log_tests.cpp
  DEPENDS_ON ${log_depends})

if (Python3_FOUND)
  add_test(
    NAME log_tests
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/log_tests_runner.py WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif ()
