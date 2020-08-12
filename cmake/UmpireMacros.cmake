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
