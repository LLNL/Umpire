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


message(STATUS "Checking for std::filesystem")

include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
  "#include <iostream>
  #include <filesystem>

  int main(int, char**)
  {

    auto path = std::filesystem::path(\".\");
    (void)(path);

    return 0;
  }"
  UMPIRE_ENABLE_FILESYSTEM)

if (UMPIRE_ENABLE_FILESYSTEM)
  message(STATUS "std::filesystem found")
else ()
  message(STATUS "std::filesystem NOT found, using POSIX")
endif ()
