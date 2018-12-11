##############################################################################
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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
find_program (BASH_PROGRAM bash)

set (replay_integration_tests_depends
  umpire)

if (BASH_PROGRAM)
  blt_add_executable(
    NAME replay_tests
    SOURCES replay_tests.cpp
    DEPENDS_ON ${replay_integration_tests_depends})

  add_test(
    NAME replay_tests 
    COMMAND ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/replay_tests.bash ${CMAKE_BINARY_DIR}/bin
  )

endif()
