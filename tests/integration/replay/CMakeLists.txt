##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
find_program (BASH_PROGRAM bash)

set (replay_integration_tests_depends
  umpire)

if (ENABLE_HIP)
  set (replay_integration_tests_depends
      ${replay_integration_tests_depends}
      hip_runtime)
endif()

if (ENABLE_OPENMP_TARGET)
  set (replay_integration_tests_depends
      ${replay_integration_tests_depends}
      openmp)
endif()

set(raw_replay_file "output" )
set(replay_config "")

if (ENABLE_NUMA)
  set(replay_config "${replay_config}numa_" )
endif()

if (ENABLE_CUDA OR ENABLE_HIP)
  set(replay_config "${replay_config}devices_" )

  if (ENABLE_DEVICE_CONST)
    set(replay_config "${replay_config}const_")
  endif ()
endif ()

if (ENABLE_OPENMP_TARGET)
  set(replay_config "${replay_config}omp_")
endif ()

set(replay_config_stripped "")
string(REGEX REPLACE "_$" "" replay_config_stripped "${replay_config}")
set(raw_replay_file "${raw_replay_file}.${replay_config_stripped}.replay")
string(REGEX REPLACE "\\.+" "." replay_file "${raw_replay_file}")

blt_add_executable(
  NAME replay_tests
  SOURCES replay_tests.cpp
  DEPENDS_ON ${replay_integration_tests_depends})

add_custom_target(
  regen_replay_output
  COMMAND  ${CMAKE_COMMAND} -E env UMPIRE_REPLAY=On $<TARGET_FILE:replay_tests> && mv *.replay ${CMAKE_CURRENT_SOURCE_DIR}/${replay_file} && rm -f *.replay)

message(STATUS "Looking for ${replay_file} reference output...")
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${replay_file})
  if (BASH_PROGRAM)
    message(STATUS "Using ${replay_file} as good replay test data")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/${replay_file}"
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_output.good"
      COPYONLY
    )

    add_test(
      NAME replay_coverage_tests
      COMMAND ${BASH_PROGRAM}
      ${CMAKE_CURRENT_SOURCE_DIR}/check_replay_coverage.bash)

    if (NOT C_COMPILER_FAMILY_IS_PGI)
      add_test(
        NAME replay_tests
        COMMAND ${BASH_PROGRAM}
        ${CMAKE_CURRENT_SOURCE_DIR}/replay_tests.bash ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endif()

  else ()

    message(STATUS "Bash not configured, replay tests disabled.")

  endif ()
else ()
  message(WARNING "No replay reference output available for current build configuration, replay test disabled.")
endif ()
