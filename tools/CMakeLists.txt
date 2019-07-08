##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

set (tools_depends umpire umpire_tpl_cxxopts umpire_tpl_json replay_util_lib)

if (ENABLE_HIP)
  set (tools_depends ${tools_depends} hip_runtime)
endif()

set (replay_util_headers
  ./util/OperationManager.hpp
  ./util/Replay.hpp)

blt_add_library(
  NAME replay_util_lib
  HEADERS ${replay_util_headers})

target_link_libraries(replay_util_lib INTERFACE)

target_include_directories(
  replay_util_lib
  INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tools>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

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

install(FILES plot_allocations DESTINATION bin)
