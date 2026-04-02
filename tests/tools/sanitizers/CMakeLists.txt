##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

include(FindPythonInterp)

if (UMPIRE_ENABLE_ASAN)
  blt_add_executable(
    NAME sanitizer_tests
    SOURCES sanitizer_tests.cpp
    DEPENDS_ON umpire)

  if (PYTHON_EXECUTABLE)
    add_test(
      NAME sanitizer_tests
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sanitizer_test_runner.py
      WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endif ()
endif ()

if (UMPIRE_ENABLE_TSAN)
  blt_add_executable(
    NAME thread_sanitizer_tests
    SOURCES thread_sanitizer_tests.cpp
    DEPENDS_ON umpire)

  if (PYTHON_EXECUTABLE)
    add_test(
      NAME thread_sanitizer_tests
      COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/thread_sanitizer_test_runner.py
      WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endif ()
endif ()
