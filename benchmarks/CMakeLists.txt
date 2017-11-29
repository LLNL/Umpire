blt_add_executable(
  NAME allocator_stress_test
  SOURCES allocator_stress_test.cpp
  DEPENDS_ON umpire
  OUTPUT_DIR ${UMPIRE_BENCHMARK_OUTPUT_DIR})

blt_add_executable(
  NAME copy_stress_test
  SOURCES copy_stress_test.cpp
  DEPENDS_ON umpire
  OUTPUT_DIR ${UMPIRE_BENCHMARK_OUTPUT_DIR})

blt_add_executable(
  NAME allocator_benchmarks
  SOURCES allocator_benchmarks.cpp
  DEPENDS_ON gbenchmark umpire
  OUTPUT_DIR ${UMPIRE_BENCHMARK_OUTPUT_DIR})

blt_add_benchmark(
  NAME allocator_benchmarks
  COMMAND allocator_benchmarks)

blt_add_executable(
  NAME copy_benchmarks
  SOURCES copy_benchmarks.cpp
  DEPENDS_ON gbenchmark umpire
  OUTPUT_DIR ${UMPIRE_BENCHMARK_OUTPUT_DIR})

blt_add_benchmark(
  NAME copy_benchmarks
  COMMAND copy_benchmarks)
