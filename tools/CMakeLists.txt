##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

# TODO why do we need the tpl depends directly here?
set(tools_depends umpire umpire_tpl_cxxopts umpire_tpl_json)

if (ENABLE_HIP)
  set(tools_depends ${tools_depends} hip_runtime)
endif()

set(replay_util_headers
  util/ReplayInterpreter.hpp
  util/ReplayInterpreter.inl
  util/ReplayMacros.hpp
  util/ReplayOperationManager.hpp
  util/ReplayOperationManager.inl)

set(replay_util_sources
  util/ReplayInterpreter.cpp
  util/ReplayOperationManager.cpp)

blt_add_executable(
  NAME replay
  SOURCES replay.cpp ${replay_util_sources}
  INCLUDES ${replay_util_headers}
  DEPENDS_ON ${tools_depends}
  OUTPUT_DIR ${TOOL_OUTPUT_DIRECTORY})
list(APPEND umpire_tools replay)

target_include_directories(replay
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tools>)

blt_add_executable(
  NAME replaydiff
  SOURCES replaydiff.cpp ${replay_util_sources}
  INCLUDES ${replay_util_headers}
  DEPENDS_ON ${tools_depends}
  OUTPUT_DIR ${TOOL_OUTPUT_DIRECTORY})
list(APPEND umpire_tools replaydiff)

target_include_directories(replaydiff
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tools>)

add_subdirectory(analysis)

install(TARGETS ${umpire_tools} DESTINATION ${UMPIRE_TOOL_INSTALL_DIRECTORY})
