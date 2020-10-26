##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
macro(umpire_add_code_checks)
    set(options)
    set(singleValueArgs PREFIX RECURSIVE)

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    if ("${PROJECT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
        set(_sources)
        if (DEFINED arg_RECURSIVE AND arg_RECURSIVE) 
          file(GLOB_RECURSE _sources
               "*.cpp" "*.hpp" "*.inl" "*.c" "*.h" "*.F" "*.f" "*.f90" "*.F90")
         else ()
           file(GLOB _sources
               "*.cpp" "*.hpp" "*.inl" "*.c" "*.h" "*.F" "*.f" "*.f90" "*.F90")
         endif ()

        blt_add_code_checks(PREFIX    ${arg_PREFIX}
                            SOURCES   ${_sources}
                            CLANGFORMAT_CFG_FILE ${PROJECT_SOURCE_DIR}/.clang-format)
    endif()
endmacro(umpire_add_code_checks)

##----------------------------------------------------------------------------
## umpire_add_test_with_mpi( NAME          [name]
##                           COMMAND       [command]
##                           NUM_MPI_TASKS [n])
##----------------------------------------------------------------------------
macro(umpire_add_test_with_mpi)

    set(options)
    set(singleValueArgs NAME NUM_MPI_TASKS)
    set(multiValueArgs COMMAND)

    ## parse the arguments to the macro
    cmake_parse_arguments(arg
     "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN} )

   if (ENABLE_MPI)
      blt_add_test( NAME           ${arg_NAME}
                    COMMAND        ${arg_COMMAND}
                    NUM_MPI_TASKS  ${arg_NUM_MPI_TASKS})
    else()
      blt_add_test( NAME           ${arg_NAME}
                    COMMAND        ${arg_COMMAND})
    endif()

endmacro(umpire_add_test_with_mpi)
