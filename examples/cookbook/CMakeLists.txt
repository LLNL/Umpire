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
set (cookbook_dir ${UMPIRE_EXAMPLES_OUTPUT_DIR}/cookbook)

blt_add_executable(
  NAME recipe_no_introspection
  SOURCES recipe_no_introspection.cpp
  DEPENDS_ON umpire
  OUTPUT_DIR ${cookbook_dir})

if (ENABLE_NUMA)
  blt_add_executable(
    NAME recipe_move_between_numa
    SOURCES recipe_move_between_numa.cpp
    DEPENDS_ON umpire
    OUTPUT_DIR ${cookbook_dir})
endif ()

if (ENABLE_CUDA)
  blt_add_executable(
    NAME recipe_shrink
    SOURCES recipe_shrink.cpp
    DEPENDS_ON umpire
    OUTPUT_DIR ${cookbook_dir})

  blt_add_executable(
    NAME recipe_advice_device_id
    SOURCES recipe_advice_device_id.cpp
    DEPENDS_ON umpire
    OUTPUT_DIR ${cookbook_dir})

  blt_add_executable(
    NAME recipe_pool_advice
    SOURCES recipe_pool_advice.cpp
    DEPENDS_ON umpire
    OUTPUT_DIR ${cookbook_dir})

  blt_add_executable(
    NAME recipe_move_to_managed
    SOURCES recipe_move_to_managed.cpp
    DEPENDS_ON umpire
    OUTPUT_DIR ${cookbook_dir})

  blt_add_executable(
    NAME recipe_pinned_pool
    SOURCES recipe_pinned_pool.F
    DEPENDS_ON umpire
    OUTPUT_DIR ${cookbook_dir})

  set_source_files_properties(
    recipe_pinned_pool.F
    PROPERTIES
    Fortran_FORMAT FREE)
endif ()

blt_add_executable(
  NAME recipe_dynamic_pool_heuristic
  SOURCES recipe_dynamic_pool_heuristic.cpp
  DEPENDS_ON umpire
  OUTPUT_DIR ${cookbook_dir})

blt_add_executable(
  NAME recipe_coalesce_pool
  SOURCES recipe_coalesce_pool.cpp
  DEPENDS_ON umpire
  OUTPUT_DIR ${cookbook_dir})
