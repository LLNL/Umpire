##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (backtrace_output_dir ${TEST_OUTPUT_DIRECTORY})

blt_add_executable(
  NAME backtrace_tests
  SOURCES backtrace_tests.cpp BacktraceTest.hpp BacktraceTest.cpp
  DEPENDS_ON umpire
  OUTPUT_DIR ${backtrace_output_dir})

blt_add_target_link_flags(
  TO backtrace_tests
  FLAGS "-rdynamic")

blt_add_target_compile_flags(
  TO backtrace_tests
  FLAGS "-O0")

if (NOT C_COMPILER_FAMILY_IS_PGI)
  include(FindPythonInterp)

  if (PYTHON_EXECUTABLE)
    add_test(
      NAME backtrace_tests
      COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/backtrace_tests_runner.py
        ${CMAKE_CURRENT_SOURCE_DIR}/backtrace.good
        WORKING_DIRECTORY ${backtrace_output_dir})
  endif ()
endif ()
