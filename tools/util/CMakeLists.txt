##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (replay_util_headers
  ReplayInterpreter.hpp
  ReplayInterpreter.inl
  ReplayOperationManager.hpp
  ReplayOperationManager.inl
)

set (replay_util_sources
  ReplayInterpreter.cpp
  ReplayOperationManager.cpp
)

blt_add_library(
  NAME replay_util
  HEADERS ${replay_util_headers}
  SOURCES ${replay_util_sources}
  OBJECT TRUE)

target_include_directories(
  replay_util
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tools>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

