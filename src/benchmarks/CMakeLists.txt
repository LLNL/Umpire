blt_add_executable(
        NAME allocation_benchmarks
        SOURCES allocation_benchmarks.cxx
        DEPENDS_ON gbenchmark umpire)

if (ENABLE_CUDA)
  blt_add_executable(
          NAME gpu_allocation_benchmarks
          SOURCES allocation_benchmarks.cxx
          DEPENDS_ON gbenchmark umpire cuda_runtime)
endif()
