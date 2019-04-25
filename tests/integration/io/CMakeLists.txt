##############################################################################
# Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
#
# Created by David Beckingsale, david@llnl.gov
# LLNL-CODE-747640
#
# All rights reserved.
#
# This file is part of Umpire.
#
# For details, see https://github.com/LLNL/Umpire
# Please also see the LICENSE file for MIT license.
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
