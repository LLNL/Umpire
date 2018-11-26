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


blt_add_library(
  NAME umpire_tpl_judy
  SOURCES ${judy_sources}
  HEADERS ${judy_headers}
  HEADERS_OUTPUT_SUBDIR umpire/tpl/judy)

if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
  target_compile_options(
    umpire_tpl_judy
    PRIVATE
    -Os)
endif()

install(TARGETS
  umpire_tpl_judy
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(FILES
  ${judy_headers}
  DESTINATION include/umpire/tpl/judy)
