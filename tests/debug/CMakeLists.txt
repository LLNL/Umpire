##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

#Intend to add more debug helper tests like this, so setting a variable specifically
set (test_debug_depends umpire)

if (ENABLE_HIP)
  set (test_debug_depends
      ${test_debug_depends}
      hip_runtime)
endif()

if (ENABLE_OPENMP_TARGET)
  set (test_debug_depends
      ${test_debug_depends}
      openmp)
endif()

if (ENABLE_BENCHMARKS)
blt_add_executable(
  NAME allocate_deallocate
  SOURCES allocate_deallocate.cpp
  DEPENDS_ON ${test_debug_depends})
list(APPEND umpire_test_debug allocate_deallocate)
endif()

install(TARGETS ${umpire_test_debug} RUNTIME DESTINATION bin)
umpire_add_code_checks(PREFIX test_debug)
blt_print_target_properties(TARGET allocate_deallocate)
