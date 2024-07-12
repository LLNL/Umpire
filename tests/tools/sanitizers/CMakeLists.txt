##############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (sanitizer_depends umpire umpire_util)

if (UMPIRE_ENABLE_HIP)
  set (sanitizer_depends
      ${sanitizer_depends}
      blt::hip)
endif ()

blt_add_executable(
  NAME sanitizer_tests
  SOURCES sanitizer_tests.cpp
  DEPENDS_ON ${sanitizer_depends})

include(FindPythonInterp)

if (PYTHON_EXECUTABLE)
  add_test(NAME sanitizer_tests
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/sanitizer_test_runner.py WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif ()
