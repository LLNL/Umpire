##############################################################################
# Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
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

if (UMPIRE_ENABLE_UMAP)
  find_library( UMAP_LIBRARY
    libumap.so
    PATHS ${UMAP_ROOT}/lib
  )
  if (NOT UMAP_LIBRARY)
    message(FATAL_ERROR "Could not find UMAP library, check UMAP installation at UMAP_ROOT")
  endif()
  find_path( UMAP_INCLUDE_DIR
    NAMES "umap/umap.h"
    PATHS ${UMAP_ROOT}/include
  )
  if (NOT UMAP_INCLUDE_DIR)
    message(FATAL_ERROR "Headers missing, check UMAP installation at UMAP_ROOT")
  endif ()
  blt_import_library(NAME umap
    INCLUDES ${UMAP_INCLUDE_DIR}
    LIBRARIES ${UMAP_LIBRARY}
    DEPENDS_ON -lpthread -lrt)
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

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  blt_register_library( NAME backtrace_symbols
    LIBRARIES ${CMAKE_DL_LIBS}
    )
endif ()

if (UMPIRE_ENABLE_SQLITE_EXPERIMENTAL)
  find_package(SQLite3 REQUIRED)
endif()

set(UMPIRE_NEEDS_BLT_TPLS False)
if (UMPIRE_ENABLE_MPI OR UMPIRE_ENABLE_HIP OR UMPIRE_ENABLE_OPENMP OR UMPIRE_ENABLE_CUDA)
  set(UMPIRE_NEEDS_BLT_TPLS True)

  if (NOT BLT_EXPORTED)
    set(BLT_EXPORTED On CACHE BOOL "" FORCE)
    blt_import_library(NAME          blt_stub EXPORTABLE On)
    set_target_properties(blt_stub PROPERTIES EXPORT_NAME blt::blt_stub)
    install(TARGETS blt_stub
      EXPORT               bltTargets)
    blt_export_tpl_targets(EXPORT bltTargets NAMESPACE blt)
    install(EXPORT bltTargets
      DESTINATION  lib/cmake/umpire)
  elseif (UMPIRE_ENABLE_MPI)
    # If the target is EXPORTABLE, add it to the export set
    get_target_property(_is_imported mpi IMPORTED)
    if(NOT ${_is_imported})
      install(TARGETS              mpi
        EXPORT               ${arg_EXPORT})
      # Namespace target to avoid conflicts
      set_target_properties(mpi PROPERTIES EXPORT_NAME blt::mpi)
      install(EXPORT bltTargets
        DESTINATION  lib/cmake/umpire)
    endif()
  endif()
endif()
