##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

blt_add_executable(
  NAME sanitizer_tests
  SOURCES sanitizer_tests.cpp
  DEPENDS_ON umpire)

include(FindPythonInterp)

if (PYTHON_EXECUTABLE)
add_test(NAME sanitizer_tests
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sanitizer_test_runner.py WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif ()