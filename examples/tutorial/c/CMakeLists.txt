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
blt_add_executable(
  NAME tut_allocator_c
  SOURCES tut_allocator.c
  DEPENDS_ON
    umpire
  OUTPUT_DIR ${tutorial_dir}
)

blt_add_executable(
  NAME tut_resources_c
  SOURCES tut_resources.c
  DEPENDS_ON
    umpire
  OUTPUT_DIR ${tutorial_dir}
)

blt_add_executable(
  NAME tut_pool_c
  SOURCES tut_pool.c
  DEPENDS_ON
    umpire
  OUTPUT_DIR ${tutorial_dir}
)
