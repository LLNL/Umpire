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

set (tools_depends umpire umpire_tpl_cxxopts umpire_tpl_json)

if (ENABLE_HIP)
  set (tools_depends ${tools_depends} hip_runtime)
endif()

blt_add_executable(
  NAME replay
  SOURCES replay.cpp
  DEPENDS_ON ${tools_depends}
  OUTPUT_DIR ${UMPIRE_TOOLS_OUTPUT_DIR})

install(FILES plot_allocations DESTINATION bin)
