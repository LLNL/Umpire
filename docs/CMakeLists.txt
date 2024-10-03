##############################################################################
# Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
  if (SPHINX_FOUND)
    blt_add_sphinx_target( umpire_docs )
    add_subdirectory(sphinx)
  endif()

  if (DOXYGEN_FOUND)
    add_subdirectory(doxygen)
  endif ()

  if ( NOT SPHINX_FOUND AND NOT DOXGEN_FOUND)
    message(WARNING "UMPIRE_ENABLE_DOCS=On, but Sphinx or Doxygen not found. \
      Documentation won't be built.")
  endif ()
