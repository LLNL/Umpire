##############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

#Intend to add more debug helper tests like this, so setting a variable specifically
set (test_debug_depends umpire)

blt_add_executable(
  NAME allocate_deallocate
  SOURCES allocate_deallocate.cpp
  DEPENDS_ON ${test_debug_depends})
list(APPEND umpire_test_debug allocate_deallocate)

if (UMPIRE_ENABLE_IPC_SHARED_MEMORY)
  blt_add_executable(
    NAME pinned_fork_repro
    SOURCES pinned_fork_repro.cpp
    DEPENDS_ON ${test_debug_depends})
  list(APPEND umpire_test_debug pinned_fork_repro)
endif()

install(TARGETS ${umpire_test_debug} RUNTIME DESTINATION bin)
umpire_add_code_checks(PREFIX test_debug)
