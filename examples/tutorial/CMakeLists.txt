##############################################################################
# Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
set (tutorial_dir ${UMPIRE_EXAMPLES_OUTPUT_DIR}/tutorial)

set (tutorial_depends umpire)

if (ENABLE_HIP)
  set (tutorial_depends
      ${tutorial_depends}
      hip)
endif()

blt_add_executable(
  NAME tut_allocator
  SOURCES tut_allocator.cpp
  DEPENDS_ON ${tutorial_depends}
  OUTPUT_DIR ${tutorial_dir})

blt_add_executable(
  NAME tut_resources
  SOURCES tut_resources.cpp
  DEPENDS_ON ${tutorial_depends}
  OUTPUT_DIR ${tutorial_dir})

blt_add_executable(
  NAME tut_dynamic_pool_1
  SOURCES tut_dynamic_pool_1.cpp
  DEPENDS_ON ${tutorial_depends}
  OUTPUT_DIR ${tutorial_dir})

blt_add_executable(
  NAME tut_dynamic_pool_2
  SOURCES tut_dynamic_pool_2.cpp
  DEPENDS_ON ${tutorial_depends}
  OUTPUT_DIR ${tutorial_dir})

blt_add_executable(
  NAME tut_copy
  SOURCES tut_copy.cpp
  DEPENDS_ON ${tutorial_depends}
  OUTPUT_DIR ${tutorial_dir})

blt_add_executable(
  NAME tut_move
  SOURCES tut_move.cpp
  DEPENDS_ON ${tutorial_depends}
  OUTPUT_DIR ${tutorial_dir})

blt_add_executable(
  NAME tut_memset
  SOURCES tut_memset.cpp
  DEPENDS_ON ${tutorial_depends}
  OUTPUT_DIR ${tutorial_dir})

blt_add_executable(
  NAME tut_reallocate
  SOURCES tut_reallocate.cpp
  DEPENDS_ON ${tutorial_depends}
  OUTPUT_DIR ${tutorial_dir})

blt_add_executable(
  NAME tut_typed_allocator
  SOURCES tut_typed_allocator.cpp
  DEPENDS_ON ${tutorial_depends}
  OUTPUT_DIR ${tutorial_dir})

blt_add_executable(
  NAME tut_introspection
  SOURCES tut_introspection.cpp
  DEPENDS_ON ${tutorial_depends}
  OUTPUT_DIR ${tutorial_dir})

blt_add_executable(
  NAME tut_replay
  SOURCES tut_replay.cpp
  DEPENDS_ON ${tutorial_depends}
  OUTPUT_DIR ${tutorial_dir})

if (ENABLE_C)
  add_subdirectory(c)
endif ()

if (ENABLE_FORTRAN)
  add_subdirectory(fortran)
endif ()
