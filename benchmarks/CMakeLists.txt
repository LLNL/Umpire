##############################################################################
# Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
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

if (ENABLE_OPENMP_TARGET)
  set (stress_test_depends
      ${stress_test_depends}
      openmp)
endif()

blt_add_executable(
  NAME allocator_stress_test
  SOURCES allocator_stress_test.cpp
  DEPENDS_ON ${stress_test_depends})

blt_add_executable(
  NAME copy_stress_test
  SOURCES copy_stress_test.cpp
  DEPENDS_ON ${stress_test_depends})

blt_add_executable(
  NAME no-op_stress_test
  SOURCES no-op_stress_test.cpp
  DEPENDS_ON ${stress_test_depends})

blt_add_benchmark(
  NAME no-op_stress_test
  COMMAND no-op_stress_test)

blt_add_executable(
  NAME pools_stress_test
  SOURCES pools_stress_test.cpp
  DEPENDS_ON ${stress_test_depends})

if (ENABLE_BENCHMARKS)
  set (benchmark_depends gbenchmark umpire)

  if (ENABLE_HIP)
    set (benchmark_depends
      ${benchmark_depends}
      hip_runtime)
  endif()

  if (ENABLE_OPENMP_TARGET)
    set (benchmark_depends
      ${benchmark_depends}
      openmp)
  endif()

  blt_add_executable(
    NAME allocator_benchmarks
    SOURCES allocator_benchmarks.cpp
    DEPENDS_ON ${benchmark_depends})

  blt_add_benchmark(
    NAME allocator_benchmarks
    COMMAND allocator_benchmarks)

  blt_add_executable(
    NAME vendor_allocator_benchmarks
    SOURCES vendor_allocator_benchmarks.cpp
    DEPENDS_ON ${benchmark_depends})

  blt_add_benchmark(
    NAME vendor_allocator_benchmarks
    COMMAND vendor_allocator_benchmarks)

  blt_add_executable(
    NAME debuglog_benchmarks
    SOURCES debuglog_benchmarks.cpp
    DEPENDS_ON ${benchmark_depends})

  if (ENABLE_OPENMP)
    set (benchmark_depends
      ${benchmark_depends}
      openmp)

    blt_add_executable(
      NAME file_resource_benchmarks
      SOURCES file_resource_benchmarks.cpp
      DEPENDS_ON ${benchmark_depends})
  endif()

  blt_add_executable(
    NAME copy_benchmarks
    SOURCES copy_benchmarks.cpp
    DEPENDS_ON ${benchmark_depends})

  blt_add_benchmark(
    NAME copy_benchmarks
    COMMAND copy_benchmarks)

  blt_add_executable(
    NAME pool_benchmarks
    SOURCES pool_benchmarks.cpp
    DEPENDS_ON ${benchmark_depends})

  blt_add_benchmark(
    NAME pool_benchmarks
    COMMAND pool_benchmarks)

  blt_add_executable(
    NAME inspector_benchmarks
    SOURCES inspector_benchmarks.cpp
    DEPENDS_ON ${benchmark_depends})

  blt_add_benchmark(
    NAME inspector_benchmarks
    COMMAND inspector_benchmarks)
endif()
