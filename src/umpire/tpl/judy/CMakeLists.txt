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
set (judy_sources
  judy.c)

set (judy_headers
  judy.h
  judyLArray.h
  judyL2Array.h)

if (ENABLE_OPENMP_TARGET)
  set (judy_depends
    openmp)
endif ()

blt_add_library(
  NAME umpire_tpl_judy
  SOURCES ${judy_sources}
  HEADERS ${judy_headers}
  DEPENDS_ON ${judy_depends}
  OBJECT TRUE)

target_include_directories(
  umpire_tpl_judy
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(FILES
  ${judy_headers}
  DESTINATION include/umpire/tpl/judy)
