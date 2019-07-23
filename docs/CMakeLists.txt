##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
if (DOXYGEN_FOUND)
  add_subdirectory(doxygen)
endif ()

if (SPHINX_FOUND)
  add_subdirectory(sphinx)
endif ()
