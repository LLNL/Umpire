##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
find_program (BASH_PROGRAM bash)

set (replay_output_dir ${TEST_OUTPUT_DIRECTORY}/integration/replay)

set (replay_integration_tests_depends
  umpire)

if (ENABLE_HIP)
  set (replay_integration_tests_depends
      ${replay_integration_tests_depends}
      hip_runtime)
endif()

if (ENABLE_CUDA)
  set(replay_file "test_output.cuda.good" )
elseif (ENABLE_HIP)
  set(replay_file "test_output.hip.good" )
else()
  set(replay_file "test_output.good" )
endif()

if (BASH_PROGRAM)

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/${replay_file}"
    "${replay_output_dir}/test_output.good"
    COPYONLY
  )

  blt_add_executable(
    NAME replay_tests
    SOURCES replay_tests.cpp
    DEPENDS_ON ${replay_integration_tests_depends}
    OUTPUT_DIR ${replay_output_dir})

  if (NOT C_COMPILER_FAMILY_IS_PGI)
    add_test(
      NAME replay_tests
      COMMAND ${BASH_PROGRAM}
      ${CMAKE_CURRENT_SOURCE_DIR}/replay_tests.bash ${replay_output_dir} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tools)
  endif()

else()

  message(STATUS, "Bash not configured, replay tests disabled.")

endif()
