##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
blt_add_executable(
  NAME tut_allocator_f
  SOURCES tut_allocator.f
  DEPENDS_ON umpire
  OUTPUT_DIR ${tutorial_output_dir})
list(APPEND umpire_tutorials tut_allocator_f)

set_source_files_properties(
  tut_allocator.f
  PROPERTIES
  Fortran_FORMAT FREE)

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
