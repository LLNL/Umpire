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
set(umpire_alloc_headers
  CudaMallocAllocator.hpp
  CudaMallocManagedAllocator.hpp
  MallocAllocator.hpp)

set (umpire_alloc_sources)

set (umpire_alloc_depends umpire_util umpire_tpl_simpool)

if (ENABLE_CUDA)
  set (umpire_alloc_headers
    ${umpire_alloc_headers}
    CnmemAllocator.hpp)

  set (umpire_alloc_depends
    ${umpire_alloc_depends}
    umpire_tpl_cnmem)
endif ()

blt_add_library(
  NAME umpire_alloc
  HEADERS ${umpire_alloc_headers}
  SOURCES ${umpire_alloc_sources}
  HEADERS_OUTPUT_SUBDIR umpire/alloc)

target_link_libraries(
  umpire_alloc
  INTERFACE
  ${umpire_alloc_depends})

install(TARGETS
  umpire_alloc
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
