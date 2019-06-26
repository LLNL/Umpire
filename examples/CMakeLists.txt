##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (example_depends umpire)

if (ENABLE_HIP)
  set (example_depends
      ${example_depends}
      hip_runtime)
endif()

blt_add_executable(
  NAME malloc
  SOURCES malloc.cxx
  DEPENDS_ON ${example_depends}
  OUTPUT_DIR ${UMPIRE_EXAMPLES_OUTPUT_DIR})

blt_add_executable(
  NAME strategy_example
  SOURCES strategy_example.cpp
  DEPENDS_ON ${example_depends}
  OUTPUT_DIR ${UMPIRE_EXAMPLES_OUTPUT_DIR})

blt_add_executable(
  NAME allocator
  SOURCES allocator.cxx
  DEPENDS_ON ${example_depends}
  OUTPUT_DIR ${UMPIRE_EXAMPLES_OUTPUT_DIR})

if (ENABLE_C)
  blt_add_executable(
    NAME allocator_c
    SOURCES allocator.c
    DEPENDS_ON ${example_depends}
    OUTPUT_DIR ${UMPIRE_EXAMPLES_OUTPUT_DIR})
endif ()

blt_add_executable(
  NAME vector_allocator
  SOURCES vector_allocator.cpp
  DEPENDS_ON ${example_depends}
  OUTPUT_DIR ${UMPIRE_EXAMPLES_OUTPUT_DIR})

blt_add_executable(
  NAME size_limiter
  SOURCES size_limiter.cpp
  DEPENDS_ON ${example_depends}
  OUTPUT_DIR ${UMPIRE_EXAMPLES_OUTPUT_DIR})

if (ENABLE_STATISTICS)
  blt_add_executable(
    NAME statistics_example
    SOURCES statistics.cpp
    DEPENDS_ON ${example_depends}
    OUTPUT_DIR ${UMPIRE_EXAMPLES_OUTPUT_DIR})
endif()

add_subdirectory(tutorial)

add_subdirectory(cookbook)

blt_add_executable(
  NAME scratch
  SOURCES scratch.cpp
  DEPENDS_ON ${example_depends})
