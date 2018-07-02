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
set (umpire_util_headers
  AllocationMap.hpp
  AllocationRecord.hpp
  Exception.hpp
  Logger.hpp
  Macros.hpp
  Platform.hpp)

set (umpire_util_sources
  AllocationMap.cpp
  Exception.cpp
  Logger.cpp)

set (umpire_util_depends umpire_tpl_judy)

if (ENABLE_SLIC AND ENABLE_LOGGING)
  set (umpire_util_depends
    ${umpire_util_depends}
    slic)
endif()

blt_add_library(
  NAME umpire_util
  HEADERS ${umpire_util_headers}
  SOURCES ${umpire_util_sources}
  HEADERS_OUTPUT_SUBDIR umpire/util
  DEPENDS_ON ${umpire_util_depends}
)

install(TARGETS
  umpire_util
  EXPORT umpire-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

