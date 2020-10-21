##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
blt_add_executable(
  NAME io_tests
  SOURCES io_tests.cpp
  DEPENDS_ON umpire)

if (NOT C_COMPILER_FAMILY_IS_PGI)
  include(FindPythonInterp)

  if (PYTHON_EXECUTABLE)
    umpire_add_test_with_mpi(
      NAME io_tests_1
      NUM_MPI_TASKS 1
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/io_tests_runner.py "Basename +logging +replay" WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    umpire_add_test_with_mpi(
      NAME io_tests_2
      NUM_MPI_TASKS 1
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/io_tests_runner.py "Basename +Directory -logging -replay" WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    umpire_add_test_with_mpi(
      NAME io_tests_3
      NUM_MPI_TASKS 1
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/io_tests_runner.py "Basename +Directory +logging -replay" WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    umpire_add_test_with_mpi(
      NAME io_tests_4
      NUM_MPI_TASKS 1
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/io_tests_runner.py "Basename +Directory -logging +replay" WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

    umpire_add_test_with_mpi(
      NAME io_tests_5
      NUM_MPI_TASKS 1
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/io_tests_runner.py "Basename +Directory +logging +replay" WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

  endif ()
endif ()

set (log_depends umpire)

if (ENABLE_HIP)
  set (log_depends
    ${log_depends}
    hip_runtime)
endif ()

if (ENABLE_OPENMP_TARGET)
  set (log_depends
    ${log_depends}
    openmp)
endif ()

blt_add_executable(
  NAME log_tests
  SOURCES log_tests.cpp
  DEPENDS_ON ${log_depends})

if (PYTHON_EXECUTABLE)
  add_test(
    NAME log_tests
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/log_tests_runner.py WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif ()
