##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

add_subdirectory(util)
add_subdirectory(analysis)

set (tools_depends umpire umpire_tpl_cxxopts umpire_tpl_json replay_util)

if (ENABLE_HIP)
  set (tools_depends ${tools_depends} hip_runtime)
endif()

blt_add_executable(
  NAME replay
  SOURCES replay.cpp
  DEPENDS_ON ${tools_depends}
  OUTPUT_DIR ${UMPIRE_TOOLS_OUTPUT_DIR})

blt_add_executable(
  NAME replaydiff
  SOURCES replaydiff.cpp
  DEPENDS_ON ${tools_depends}
  OUTPUT_DIR ${UMPIRE_TOOLS_OUTPUT_DIR})
