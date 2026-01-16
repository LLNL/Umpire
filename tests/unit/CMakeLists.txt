##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (unit_depends umpire gtest)

if (UMPIRE_ENABLE_OPENMP_TARGET)
  set (unit_depends ${unit_depends} openmp)
endif ()

blt_add_executable(
  NAME resource_manager_tests
  SOURCES resource_manager_tests.cpp
  DEPENDS_ON ${unit_depends})

blt_add_test(
  NAME resource_manager_tests
  COMMAND resource_manager_tests)

blt_add_executable(
  NAME version_tests
  SOURCES version_tests.cpp
  DEPENDS_ON ${unit_depends})

blt_add_test(
  NAME version_tests
  COMMAND version_tests)

blt_add_executable(
  NAME umpire_tests
  SOURCES umpire_tests.cpp
  DEPENDS_ON ${unit_depends})

blt_add_test(
  NAME umpire_tests
  COMMAND umpire_tests)

blt_add_executable(
  NAME api_v2_registry_tests
  SOURCES api_v2/test_registry.cpp
  DEPENDS_ON ${unit_depends})

blt_add_test(
  NAME api_v2_registry_tests
  COMMAND api_v2_registry_tests)

blt_add_executable(
  NAME api_v2_memory_tests
  SOURCES api_v2/test_memory.cpp
  DEPENDS_ON ${unit_depends})

blt_add_test(
  NAME api_v2_memory_tests
  COMMAND api_v2_memory_tests)

add_subdirectory(alloc)
add_subdirectory(op)
add_subdirectory(resource)
add_subdirectory(util)
add_subdirectory(strategy)
