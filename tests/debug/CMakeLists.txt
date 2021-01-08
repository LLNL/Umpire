##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

#Intend to add more debug helper tests like this, so setting a variable specifically
set (test_debug_depends umpire)

if (ENABLE_BENCHMARKS)
blt_add_executable(
  NAME allocate_deallocate
  SOURCES allocate_deallocate.cpp
  DEPENDS_ON ${test_debug_depends})
endif()

