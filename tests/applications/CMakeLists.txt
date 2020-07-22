##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (test_applications_depends umpire)

blt_add_executable(
  NAME pool_allocator_stress
  SOURCES pool_allocator_stress.cpp
  DEPENDS_ON ${test_applications_depends})
list(APPEND umpire_test_applications pool_allocator_stress)

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  blt_add_target_link_flags(
    TO pool_allocator_stress
    FLAGS "-rdynamic -l${CMAKE_DL_LIBS}")
endif ()

install(TARGETS ${umpire_test_applications} RUNTIME DESTINATION bin)
