blt_add_executable(
  NAME allocation_benchmarks
  SOURCES allocation_benchmarks.cxx
  DEPENDS_ON gbenchmark umpire)

blt_add_benchmark(
  NAME allocation_benchmarks
  COMMAND allocation_benchmarks)
  
blt_add_executable(
  NAME pool_benchmarks
  SOURCES pool_benchmarks.cpp
  DEPENDS_ON gbenchmark umpire)

blt_add_benchmark(
  NAME pool_benchmarks
  COMMAND pool_benchmarks)

if (ENABLE_CUDA)
  blt_add_executable(
    NAME gpu_allocation_benchmarks
    SOURCES gpu_allocation_benchmarks.cxx
    DEPENDS_ON gbenchmark umpire_alloc cuda_runtime)

  blt_add_benchmark(
    NAME gpu_allocation_benchmarks
    COMMAND gpu_allocation_benchmarks)

endif()
