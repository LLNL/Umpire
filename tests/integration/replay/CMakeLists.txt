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

set(replay_file "test_output" )

if (ENABLE_NUMA OR ENABLE_CUDA)
  set(sys_type $ENV{SYS_TYPE})
  if (sys_type)
    set(replay_file "${replay_file}.${sys_type}")
  endif ()
endif ()

if (ENABLE_NUMA)
  set(replay_file "${replay_file}.numa" )
endif()

if (ENABLE_CUDA OR ENABLE_HIP)
  set(replay_file "${replay_file}.devices" )

  if (ENABLE_DEVICE_CONST)
    set(replay_file "${replay_file}.const")
  endif ()
endif ()

set(replay_file "${replay_file}.good")

blt_add_executable(
  NAME replay_tests
  SOURCES replay_tests.cpp
  DEPENDS_ON ${replay_integration_tests_depends})

add_custom_target(
  regen_replay_output
  COMMAND  ${CMAKE_COMMAND} -E env UMPIRE_REPLAY=On $<TARGET_FILE:replay_tests> && mv *.replay ${replay_file}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${replay_file})
  if (BASH_PROGRAM)
    message(STATUS "Using ${replay_file} as good replay test data")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/${replay_file}"
      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_output.good"
      COPYONLY
    )

    if (NOT C_COMPILER_FAMILY_IS_PGI)
      add_test(
        NAME replay_tests
        COMMAND ${BASH_PROGRAM}
        ${CMAKE_CURRENT_SOURCE_DIR}/replay_tests.bash ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    endif()

  else ()

    message(STATUS, "Bash not configured, replay tests disabled.")

  endif ()
else ()
  message(STATUS, "No replay reference output available for current build configuration, replay test disabled.")
endif ()
