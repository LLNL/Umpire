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

if (ENABLE_STATISTICS)
  set(OLD_COPY_HEADERS ${ENABLE_COPY_HEADERS})
  set(OLD_ENABLE_TESTS ${ENABLE_TESTS})
  set(ENABLE_COPY_HEADERS Off)
  set(ENABLE_TESTS Off)
  add_subdirectory(conduit/src)
  set(ENABLE_COPY_HEADERS ${OLD_COPY_HEADERS})
  set(ENABLE_ENABLE_TESTS ${OLD_ENABLE_TESTS})

  set (conduit_includes 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/conduit/src/libs/conduit> 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/conduit/src/libs/conduit/c>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/conduit/src/libs/conduit/python>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/conduit/src/libs/conduit>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/conduit/src/libs/conduit/python>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>)

  blt_register_library(
    NAME conduit
    INCLUDES ${conduit_includes}
    LIBRARIES conduit)
endif()

add_subdirectory(simpool)
add_subdirectory(judy)
