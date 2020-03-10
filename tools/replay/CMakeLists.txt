##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

set(tools_depends umpire umpire_tpl_json)

if (ENABLE_HIP)
  set(tools_depends ${tools_depends} hip_runtime)
endif()

set(replay_headers
  ReplayInterpreter.hpp
  ReplayInterpreter.inl
  ReplayMacros.hpp
  ReplayOperationManager.hpp
  ReplayFile.hpp)

set(replay_sources
  ReplayInterpreter.cpp
  ReplayOperationManager.cpp
  ReplayFile.cpp)

blt_add_executable(
  NAME replay
  SOURCES replay.cpp ${replay_sources}
  DEPENDS_ON ${tools_depends})
list(APPEND replay_tools replay)

blt_add_executable(
  NAME replaydiff
  SOURCES replaydiff.cpp ${replay_sources}
  DEPENDS_ON ${tools_depends})
list(APPEND replay_tools replaydiff)

install(TARGETS ${replay_tools} RUNTIME DESTINATION bin)
