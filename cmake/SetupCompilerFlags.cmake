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
if (ENABLE_COVERAGE)
  message(INFO "Coverage analysis enabled")
  set(CMAKE_CXX_FLAGS "-coverage ${CMAKE_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "-coverage ${CMAKE_EXE_LINKER_FLAGS}")
endif ()

if (ENABLE_HIP)
	set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -Wno-inconsistent-missing-override")
endif()

blt_append_custom_compiler_flag(
  FLAGS_VAR UMPIRE_PEDANTIC_FLAG
  DEFAULT  "-Wpedantic"
  MSVC "/Wall /WX"
)

set(CMAKE_CXX_FLAGS "${UMPIRE_PEDANTIC_FLAG} ${CMAKE_CXX_FLAGS}")
