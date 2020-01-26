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

set(replay_file "test_output" )

if (ENABLE_NUMA)
  set(replay_file "${replay_file}.numa" )
endif()

if (ENABLE_CUDA)
  set(replay_file "${replay_file}.cuda" )
elseif (ENABLE_HIP)
  set(replay_file "${replay_file}.hip" )
endif()

set(replay_file "${replay_file}.good" )

message(STATUS "Using ${replay_file} as good replay test data")

if (BASH_PROGRAM)

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/${replay_file}"
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_output.good"
    COPYONLY
  )

  blt_add_executable(
    NAME replay_tests
    SOURCES replay_tests.cpp
    DEPENDS_ON ${replay_integration_tests_depends})

  if (NOT C_COMPILER_FAMILY_IS_PGI)
    add_test(
      NAME replay_tests
      COMMAND ${BASH_PROGRAM}
      ${CMAKE_CURRENT_SOURCE_DIR}/replay_tests.bash ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  endif()

else()

  message(STATUS, "Bash not configured, replay tests disabled.")

endif()
