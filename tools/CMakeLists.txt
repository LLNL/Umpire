##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

install(
  FILES analysis/plot_allocations analysis/plot_allocator_traces
  DESTINATION ${CMAKE_INSTALL_BINDIR})

add_subdirectory(replay)
