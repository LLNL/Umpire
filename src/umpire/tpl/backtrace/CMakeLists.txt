##############################################################################
# Copyright (c) 2018, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
#
# Created by David Beckingsale, david@llnl.gov
# LLNL-CODE-747640
#
# All rights reserved.
#
# This file is part of Umpire.
#
# For details, see https://github.com/LLNL/Umpire
# Please also see the LICENSE file for MIT license.
##############################################################################
set (BackTrace_sources
  BackTrace.cxx)

set (BackTrace_headers
  BackTrace.hpp)

blt_add_library(
  NAME umpire_tpl_backtrace
  SOURCES ${BackTrace_sources}
  HEADERS ${BackTrace_headers}
  OBJECT TRUE)

target_include_directories(
  umpire_tpl_backtrace
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(FILES
  ${BackTrace_headers}
  DESTINATION include/umpire/tpl/BackTrace)
