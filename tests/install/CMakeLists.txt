##############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################

configure_file(
    using-with-cmake/host-config.cmake.in
    ${PROJECT_BINARY_DIR}/examples/using-with-cmake/host-config.cmake)

install( FILES
  using-with-cmake/CMakeLists.txt
  using-with-cmake/using-with-cmake.cpp
  ${PROJECT_BINARY_DIR}/examples/using-with-cmake/host-config.cmake
  DESTINATION examples/umpire/using-with-cmake)
