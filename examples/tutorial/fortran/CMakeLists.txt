##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
blt_add_executable(
  NAME tut_allocator_fortran
  SOURCES tut_allocator.F
  DEPENDS_ON umpire hip_runtime)
list(APPEND umpire_tutorials tut_allocator_fortran)

set_source_files_properties(
  tut_allocator.F
  PROPERTIES
  Fortran_FORMAT FREE)

if(ENABLE_HIP)
  set_target_properties(tut_allocator_fortran PROPERTIES LINKER_LANGUAGE HIP)
endif()

# blt_add_executable(
#   NAME tut_resources_f
#   SOURCES tut_resources.f
#   DEPENDS_ON
#     umpire
#   OUTPUT_DIR ${tutorial_dir}
# )
#
# blt_add_executable(
#   NAME tut_pool_f
#   SOURCES tut_pool.f
#   DEPENDS_ON
#     umpire
#   OUTPUT_DIR ${tutorial_dir}
# )
