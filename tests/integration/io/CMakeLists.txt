##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (io_dir ${UMPIRE_TESTS_OUTPUT_DIR}/integration/io)

blt_add_executable(
  NAME iomanager_tests
  SOURCES iomanager_tests.cpp
  DEPENDS_ON umpire
  OUTPUT_DIR ${io_dir})

include(FindPythonInterp)

if (PYTHON_EXECUTABLE)
  add_test(
    NAME iomanager_tests
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/iomanager_tests_runner.py WORKING_DIRECTORY ${io_dir})
endif ()
