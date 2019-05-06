##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (cookbook_dir ${UMPIRE_EXAMPLES_OUTPUT_DIR}/cookbook)

set (cookbook_depends umpire)

if (ENABLE_HIP)
  set (cookbook_depends ${cookbook_depends} hip_runtime)
endif()

blt_add_executable(
  NAME recipe_no_introspection
  SOURCES recipe_no_introspection.cpp
  DEPENDS_ON ${cookbook_depends}
  OUTPUT_DIR ${cookbook_dir})

if (ENABLE_SICM)
  blt_add_executable(
    NAME recipe_move_between_sicm_devices
    SOURCES recipe_move_between_sicm_devices.cpp
    DEPENDS_ON umpire
    OUTPUT_DIR ${cookbook_dir})
endif ()

if (ENABLE_NUMA)
  blt_add_executable(
    NAME recipe_move_between_numa
    SOURCES recipe_move_between_numa.cpp
    DEPENDS_ON ${cookbook_depends}
    OUTPUT_DIR ${cookbook_dir})
endif ()

if (ENABLE_CUDA)
  blt_add_executable(
    NAME recipe_shrink
    SOURCES recipe_shrink.cpp
    DEPENDS_ON ${cookbook_depends}
    OUTPUT_DIR ${cookbook_dir})

  blt_add_executable(
    NAME recipe_advice_device_id
    SOURCES recipe_advice_device_id.cpp
    DEPENDS_ON ${cookbook_depends}
    OUTPUT_DIR ${cookbook_dir})

  blt_add_executable(
    NAME recipe_pool_advice
    SOURCES recipe_pool_advice.cpp
    DEPENDS_ON ${cookbook_depends}
    OUTPUT_DIR ${cookbook_dir})

  blt_add_executable(
    NAME recipe_move_to_managed
    SOURCES recipe_move_to_managed.cpp
    DEPENDS_ON ${cookbook_depends}
    OUTPUT_DIR ${cookbook_dir})

  if (ENABLE_FORTRAN)
    blt_add_executable(
      NAME recipe_pinned_pool
      SOURCES recipe_pinned_pool.F
      DEPENDS_ON ${cookbook_depends}
      OUTPUT_DIR ${cookbook_dir})

    set_source_files_properties(
      recipe_pinned_pool.F
      PROPERTIES
      Fortran_FORMAT FREE)
  endif ()
endif ()

if (ENABLE_HIP)
  blt_add_executable(
    NAME recipe_shrink
    SOURCES recipe_shrink.cpp
    DEPENDS_ON ${cookbook_depends}
    OUTPUT_DIR ${cookbook_dir})

  if (ENABLE_FORTRAN)
    blt_add_executable(
      NAME recipe_pinned_pool
      SOURCES recipe_pinned_pool.F
      DEPENDS_ON ${cookbook_depends}
      OUTPUT_DIR ${cookbook_dir})

    set_source_files_properties(
      recipe_pinned_pool.F
      PROPERTIES
      Fortran_FORMAT FREE)
  endif ()
endif ()


blt_add_executable(
  NAME recipe_dynamic_pool_heuristic
  SOURCES recipe_dynamic_pool_heuristic.cpp
  DEPENDS_ON ${cookbook_depends}
  OUTPUT_DIR ${cookbook_dir})

blt_add_executable(
  NAME recipe_coalesce_pool
  SOURCES recipe_coalesce_pool.cpp
  DEPENDS_ON ${cookbook_depends}
  OUTPUT_DIR ${cookbook_dir})

blt_add_executable(
  NAME recipe_plot_allocations
  SOURCES recipe_plot_allocations.cpp
  DEPENDS_ON ${cookbook_depends}
  OUTPUT_DIR ${cookbook_dir})

blt_add_executable(
  NAME recipe_mixed_pool
  SOURCES recipe_mixed_pool.cpp
  DEPENDS_ON ${cookbook_depends}
  OUTPUT_DIR ${cookbook_dir})
