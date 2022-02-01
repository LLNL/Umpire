##############################################################################
# Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
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
  if(NOT DEFINED ENV{UMAP_ROOT})
    message(FATAL_ERROR "UMAP_ROOT undefined")
  endif ()
  find_path( UMAP_INCLUDE_DIR
    NAMES "umap/umap.h"
    PATHS ($ENV{UMAP_ROOT}/install/include)
  )
  find_library( UMAP_LIBRARY
    libumap.a
    PATHS ($ENV{UMAP_ROOT}/install/lib)
  )
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

set(TPL_DEPS)
blt_list_append(TO TPL_DEPS ELEMENTS cuda cuda_runtime IF ENABLE_CUDA)
blt_list_append(TO TPL_DEPS ELEMENTS blt_hip blt_hip_runtime IF ENABLE_HIP)
blt_list_append(TO TPL_DEPS ELEMENTS openmp IF ENABLE_OPENMP)
blt_list_append(TO TPL_DEPS ELEMENTS mpi IF ENABLE_MPI)
blt_list_append(TO TPL_DEPS ELEMENTS umap IF ENABLE_UMAP)

foreach(dep ${TPL_DEPS})
    # If the target is EXPORTABLE, add it to the export set
    get_target_property(_is_imported ${dep} IMPORTED)
    if(NOT ${_is_imported})
        install(TARGETS              ${dep}
                EXPORT               umpire-targets
                DESTINATION          lib)
        # Namespace target to avoid conflicts
        set_target_properties(${dep} PROPERTIES EXPORT_NAME umpire::${dep})
    endif()
endforeach()
