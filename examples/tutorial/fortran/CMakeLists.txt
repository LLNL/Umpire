##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
if(ENABLE_HIP)
  blt_add_executable(
    NAME tut_allocator_fortran
    SOURCES tut_allocator.F
    DEPENDS_ON umpire hip_runtime)
  list(APPEND umpire_tutorials tut_allocator_fortran)

  set_source_files_properties(
    tut_allocator.F
    PROPERTIES
    Fortran_FORMAT FREE)

  #Until more stable blt changes can be made, this is a temporary fix to allow
  #the fortran program to be linked with hip
  set_target_properties(tut_allocator_fortran PROPERTIES LINKER_LANGUAGE HIP)
endif()

if(ENABLE_CUDA)
  blt_add_executable(
    NAME tut_allocator_fortran
    SOURCES tut_allocator.F
    DEPENDS_ON umpire cuda_runtime)
  list(APPEND umpire_tutorials tut_allocator_fortran)

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
