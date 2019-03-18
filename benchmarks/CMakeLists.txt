##############################################################################
# Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
#
# Created by David Beckingsale, david@llnl.gov
# LLNL-CODE-747640
#
# All rights reserved.
#
# This file is part of Umpire.
#
# For details, see https://github.com/LLNL/Umpire
# Please also see the LICENSE file for MIT license.
##############################################################################
blt_add_executable(
  NAME allocator_stress_test
  SOURCES allocator_stress_test.cpp
  DEPENDS_ON umpire
  OUTPUT_DIR ${UMPIRE_BENCHMARKS_OUTPUT_DIR})

blt_add_executable(
  NAME copy_stress_test
  SOURCES copy_stress_test.cpp
  DEPENDS_ON umpire
  OUTPUT_DIR ${UMPIRE_BENCHMARKS_OUTPUT_DIR})

blt_add_executable(
  NAME allocator_benchmarks
  SOURCES allocator_benchmarks.cpp
  DEPENDS_ON gbenchmark umpire
  OUTPUT_DIR ${UMPIRE_BENCHMARKS_OUTPUT_DIR})

blt_add_executable(
  NAME debuglog_benchmarks
  SOURCES debuglog_benchmarks.cpp
  DEPENDS_ON gbenchmark umpire
  OUTPUT_DIR ${UMPIRE_BENCHMARKS_OUTPUT_DIR})

blt_add_benchmark(
  NAME allocator_benchmarks
  COMMAND allocator_benchmarks)

blt_add_executable(
  NAME copy_benchmarks
  SOURCES copy_benchmarks.cpp
  DEPENDS_ON gbenchmark umpire
  OUTPUT_DIR ${UMPIRE_BENCHMARKS_OUTPUT_DIR})

blt_add_benchmark(
  NAME copy_benchmarks
  COMMAND copy_benchmarks)
