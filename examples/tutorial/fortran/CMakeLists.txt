##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (tutorial_fortran_depends umpire)

if (ENABLE_HIP)
  set (tutorial_fortran_depends
      ${tutorial_fortran_depends}
      hip_runtime)
endif()

if(ENABLE_CUDA)
  set (tutorial_fortran_depends
      ${tutorial_fortran_depends}
      cuda_runtime)
endif()

blt_add_executable(
  NAME tut_allocator_fortran
  SOURCES tut_allocator.F
  DEPENDS_ON ${tutorial_fortran_depends})
list(APPEND umpire_tutorials tut_allocator_fortran)

set_source_files_properties(
  tut_allocator.F
  PROPERTIES
  Fortran_FORMAT FREE)

  #Until more stable blt changes can be made, this is a temporary fix to allow
  #the fortran program to be linked with hip (https://rzlc.llnl.gov/jira/browse/UM-790)
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
