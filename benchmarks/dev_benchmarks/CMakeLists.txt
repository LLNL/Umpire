##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (dev_stress_test_depends umpire)

if (ENABLE_HIP)
  set (dev_stress_test_depends
      ${dev_stress_test_depends}
      hip_runtime)
endif()

if (ENABLE_OPENMP_TARGET)
  set (dev_stress_test_depends
      ${dev_stress_test_depends}
      openmp)
endif()

blt_add_executable(
  NAME no-op_stress_test
  SOURCES no-op_stress_test.cpp
  DEPENDS_ON ${dev_stress_test_depends})

blt_add_benchmark(
  NAME no-op_stress_test
  COMMAND no-op_stress_test)
