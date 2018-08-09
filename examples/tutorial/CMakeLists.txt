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

blt_add_executable(
  NAME tut_allocator
  SOURCES tut_allocator.cpp
  DEPENDS_ON 
    umpire)

blt_add_executable(
  NAME tut_resources
  SOURCES tut_resources.cpp
  DEPENDS_ON 
    umpire)

blt_add_executable(
  NAME tut_dynamic_pool
  SOURCES tut_dynamic_pool.cpp
  DEPENDS_ON 
    umpire)

blt_add_executable(
  NAME tut_copy
  SOURCES tut_copy.cpp
  DEPENDS_ON 
    umpire)

blt_add_executable(
  NAME tut_move
  SOURCES tut_move.cpp
  DEPENDS_ON 
    umpire)
