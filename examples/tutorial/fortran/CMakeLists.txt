##############################################################################
# Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (tutorial_fortran_depends umpire)

if(UMPIRE_ENABLE_CUDA)
  set (tutorial_fortran_depends
      ${tutorial_fortran_depends}
      cuda_runtime)
endif()

blt_add_executable(
  NAME tut_allocator_fortran
  SOURCES tut_allocator.F
  DEPENDS_ON ${tutorial_fortran_depends})
list(APPEND umpire_tutorials tut_allocator_fortran)

if (Fortran_COMPILER_FAMILY_IS_CLANG)
  set_source_files_properties(
    tut_allocator.F
    PROPERTIES
    COMPILE_FLAGS -Mfree)
else()
  set_source_files_properties(
    tut_allocator.F
    PROPERTIES
    Fortran_FORMAT FREE)
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
