##############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
if((NOT DEFINED CMAKE_C_STANDARD) OR (CMAKE_C_STANDARD VERSION_EQUAL 90))
    message(STATUS "Setting C standard to 99")
    set(CMAKE_C_STANDARD 99)
endif()

message(STATUS "Checking for std::filesystem")

blt_check_code_compiles(CODE_COMPILES UMPIRE_ENABLE_FILESYSTEM
                        SOURCE_STRING
[=[
#include <iostream>
#include <filesystem>

int main(int, char**)
{
    auto path = std::filesystem::path(\".\");
    (void)(path);

    return 0;
}
]=])

if (UMPIRE_ENABLE_FILESYSTEM)
  message(STATUS "std::filesystem found")
else ()
  message(STATUS "std::filesystem NOT found, using POSIX")
endif ()

if (ENABLE_HIP)
  set(HIP_HIPCC_FLAGS "${HIP_HIPCC_FLAGS} -Wno-inconsistent-missing-override")
endif()

if (ENABLE_PEDANTIC_WARNINGS)
  blt_append_custom_compiler_flag(
    FLAGS_VAR UMPIRE_PEDANTIC_FLAG
    DEFAULT  ""
    GNU "-Wpedantic"
    CLANG "-Wpedantic"
    INTEL "-Wall -Wcheck -wd2259"
    XL "-Wpedantic"
    MSVC "/Wall /WX"
  )

  set(CMAKE_CXX_FLAGS "${UMPIRE_PEDANTIC_FLAG} ${CMAKE_CXX_FLAGS}")

  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  endif()
endif()

blt_append_custom_compiler_flag(
  FLAGS_VAR UMPIRE_DISABLE_DEPRECATED_WARNINGS_FLAG
  DEFAULT  "-Wno-deprecated-declarations"
  PGI "-Minform=severe"
  INTEL "-diag-disable=1786"
  MSVC "/wd4996")
