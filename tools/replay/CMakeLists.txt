##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

set(tools_depends umpire)

if (ENABLE_HIP)
  set(tools_depends ${tools_depends} hip_runtime)
endif()

set(replay_headers
  ReplayInterpreter.hpp
  ReplayInterpreter.inl
  ReplayMacros.hpp
  ReplayOperationManager.hpp
  ReplayOperationManager.inl)

set(replay_sources
  ReplayInterpreter.cpp
  ReplayOperationManager.cpp)

blt_add_executable(
  NAME replay
  SOURCES replay.cpp ${replay_sources}
  DEPENDS_ON ${tools_depends}
  OUTPUT_DIR ${TOOL_OUTPUT_DIRECTORY})
list(APPEND replay_tools replay)

blt_add_executable(
  NAME replaydiff
  SOURCES replaydiff.cpp ${replay_sources}
  DEPENDS_ON ${tools_depends}
  OUTPUT_DIR ${TOOL_OUTPUT_DIRECTORY})
list(APPEND replay_tools replaydiff)

install(TARGETS ${replay_tools} DESTINATION ${UMPIRE_TOOL_INSTALL_DIRECTORY})
