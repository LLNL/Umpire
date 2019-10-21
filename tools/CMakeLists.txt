##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

install(
  FILES analysis/plot_allocations analysis/plot_allocator_traces
  DESTINATION ${UMPIRE_TOOL_INSTALL_DIRECTORY})

add_subdirectory(replay)
