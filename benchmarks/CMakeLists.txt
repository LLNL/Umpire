##############################################################################
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
# project contributors. See the COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)
##############################################################################
set (stress_test_depends umpire)

if (ENABLE_HIP)
  set (stress_test_depends
      ${stress_test_depends}
      hip_runtime)
endif()

blt_add_executable(
  NAME allocator_stress_test
  SOURCES allocator_stress_test.cpp
  DEPENDS_ON ${stress_test_depends}
  OUTPUT_DIR ${UMPIRE_BENCHMARKS_OUTPUT_DIR})

blt_add_executable(
  NAME copy_stress_test
  SOURCES copy_stress_test.cpp
  DEPENDS_ON ${stress_test_depends}
  OUTPUT_DIR ${UMPIRE_BENCHMARKS_OUTPUT_DIR})

set (benchmark_depends gbenchmark umpire)

if (ENABLE_HIP)
  set (benchmark_depends
      ${benchmark_depends}
      hip_runtime)
endif()

blt_add_executable(
  NAME allocator_benchmarks
  SOURCES allocator_benchmarks.cpp
  DEPENDS_ON ${benchmark_depends}
  OUTPUT_DIR ${UMPIRE_BENCHMARKS_OUTPUT_DIR})

configure_file(allocator_relative_benchmarks allocator_relative_benchmarks COPYONLY)

blt_add_executable(
  NAME debuglog_benchmarks
  SOURCES debuglog_benchmarks.cpp
  DEPENDS_ON ${benchmark_depends}
  OUTPUT_DIR ${UMPIRE_BENCHMARKS_OUTPUT_DIR})

blt_add_benchmark(
  NAME allocator_benchmarks
  COMMAND allocator_benchmarks)

blt_add_executable(
  NAME copy_benchmarks
  SOURCES copy_benchmarks.cpp
  DEPENDS_ON ${benchmark_depends}
  OUTPUT_DIR ${UMPIRE_BENCHMARKS_OUTPUT_DIR})

blt_add_benchmark(
  NAME copy_benchmarks
  COMMAND copy_benchmarks)

blt_add_executable(
  NAME inspector_benchmarks
  SOURCES inspector_benchmarks.cpp
  DEPENDS_ON ${benchmark_depends}
  OUTPUT_DIR ${UMPIRE_BENCHMARKS_OUTPUT_DIR})
