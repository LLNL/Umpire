##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
if(ENABLE_TESTS)
    set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full --gen-suppressions=all")
    file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}/cmake/valgrind.supp" MEMORYCHECK_SUPPRESSIONS_FILE)
    include(CTest)
    message(STATUS "Memcheck suppressions file: ${MEMORYCHECK_SUPPRESSIONS_FILE}")
endif()

include(cmake/UmpireMacros.cmake)
