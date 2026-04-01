##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
blt_add_executable(
  NAME api_v2_stl_containers
  SOURCES stl_containers.cpp
  DEPENDS_ON ${example_depends})

target_include_directories(
  api_v2_stl_containers
  BEFORE PRIVATE
  ${PROJECT_SOURCE_DIR}/include)

install(TARGETS api_v2_stl_containers RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
