#.rst:
# FindSLIC
# -------
#
# Find libslic, the official reference library for SLIC logging
#
# This module will set the following variables in your project:
#
# ``SLIC_LIBRARY``
#   the library to link against to use SLIC.
# ``SLIC_FOUND``
#   If false, do not try to use SLIC.
# ``SLIC_INCLUDE_DIR``
#   where to find slic.h, etc.

# ``SLIC_DEFINITIONS``
#   You should add_definitions(${SLIC_DEFINITIONS}) before compiling code
#   that includes slic library files.
#
find_library( SLIC_LIBRARY
  libslic.a
  PATHS ${SLIC_LIBRARY_PATH} 
)

find_path( SLIC_INCLUDE_DIR slic/slic.hpp
  PATHS ${SLIC_INCLUDE_PATH}
)

if (NOT SLIC_LIBRARY)
  message(FATAL_ERROR "Could not find SLIC library, make sure SLIC_LIBRARY_PATH is set properly")
endif()

if (NOT SLIC_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find SLIC include directory, make sure SLIC_INCLUDE_PATH is set properly")
endif()

set(SLIC_DEFINITIONS -DUSE_SLIC_FOR_UMPIRE_LOG)
set(SLIC_FOUND)
