##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

set(tools_depends umpire umpire_tpl_cxxopts umpire_tpl_json replay_util)

if (ENABLE_HIP)
  set(tools_depends ${tools_depends} hip_runtime)
endif()

blt_add_executable(
  NAME replay
  SOURCES replay.cpp
  DEPENDS_ON ${tools_depends}
  OUTPUT_DIR ${TOOL_OUTPUT_DIRECTORY})
list(APPEND umpire_tools replay)

target_include_directories(replay
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tools>)

blt_add_executable(
  NAME replaydiff
  SOURCES replaydiff.cpp
  DEPENDS_ON ${tools_depends}
  OUTPUT_DIR ${TOOL_OUTPUT_DIRECTORY})

target_include_directories(replaydiff
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tools>)

list(APPEND umpire_tools replaydiff)

add_subdirectory(util)
add_subdirectory(analysis)

install(TARGETS ${umpire_tools} DESTINATION ${UMPIRE_TOOL_INSTALL_DIRECTORY})
