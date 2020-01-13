##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
if (EXISTS ${SHROUD_EXECUTABLE})
  execute_process(COMMAND ${SHROUD_EXECUTABLE}
    --cmake ${CMAKE_CURRENT_BINARY_DIR}/SetupShroud.cmake
    ERROR_VARIABLE SHROUD_cmake_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (${SHROUD_cmake_error})
    message(FATAL_ERROR "Error from Shroud: ${SHROUD_cmake_error}")
  endif ()
  include(${CMAKE_CURRENT_BINARY_DIR}/SetupShroud.cmake)
endif ()

if (ENABLE_SLIC AND ENABLE_LOGGING)
  find_library( SLIC_LIBRARY
    libslic.a
    PATHS ${SLIC_LIBRARY_PATH} 
  )

  if (NOT SLIC_LIBRARY)
    message(FATAL_ERROR "Could not find SLIC library, make sure SLIC_LIBRARY_PATH is set properly")
  endif()

  find_library( SLIC_UTIL_LIBRARY
    libaxom_utils.a
    PATHS ${SLIC_LIBRARY_PATH} 
  )

  if (NOT SLIC_UTIL_LIBRARY)
    message(FATAL_ERROR "Could not find Axom Utility Library for SLIC, make sure SLIC_LIBRARY_PATH is set properly")
  endif()

  find_path( SLIC_INCLUDE_DIR
    slic/slic.hpp
    PATHS ${SLIC_INCLUDE_PATH}
  )

  if (NOT SLIC_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find SLIC include directory, make sure SLIC_INCLUDE_PATH is set properly")
  endif()

  blt_register_library( NAME slic
                        INCLUDES ${SLIC_INCLUDE_DIR}
                        LIBRARIES ${SLIC_LIBRARY} ${SLIC_UTIL_LIBRARY}
                      )
endif ()
