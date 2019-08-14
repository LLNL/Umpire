##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (io_dir ${UMPIRE_TESTS_OUTPUT_DIR}/integration/io)

blt_add_executable(
  NAME io_tests
  SOURCES io_tests.cpp
  DEPENDS_ON umpire
  OUTPUT_DIR ${io_dir})

if (NOT C_COMPILER_FAMILY_IS_PGI)
  include(FindPythonInterp)

  if (PYTHON_EXECUTABLE)
    add_test(
      NAME io_tests
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/io_tests_runner.py WORKING_DIRECTORY ${io_dir})
  endif ()
endif ()

set (log_depends umpire)

if (ENABLE_HIP)
  set (log_depends
    ${log_depends}
    hip_runtime)
endif ()

blt_add_executable(
  NAME log_tests
  SOURCES log_tests.cpp
  DEPENDS_ON ${log_depends}
  OUTPUT_DIR ${io_dir})

if (PYTHON_EXECUTABLE)
  add_test(
    NAME log_tests
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/log_tests_runner.py WORKING_DIRECTORY ${io_dir})
endif ()
