# API v2 Benchmark Coverage

This repository now contains a host-focused API v2 benchmark slice that is
safe to build and run on this machine.

Covered targets:

- `api_v2_host_memory_benchmarks`
  - raw host allocation baseline
  - API v2 host allocation with tracking on and off
- `api_v2_fixed_pool_benchmarks`
  - fixed-pool allocation and churn patterns
- `api_v2_allocator_benchmarks`
  - STL `vector::resize()` with `std::allocator`, v1 `TypedAllocator`, and
    v2 tracked/untracked allocators
  - `std::map` insertion with the same allocator comparison
  - layered strategy overhead for direct host, named host,
    `thread_safe<named<...>>`, and `thread_safe<fixed_pool<...>>`
  - compile-time host dispatch sanity checks against a direct host loop

LLVM 19 macOS configuration:

```bash
cmake -S . -B build-codex-2m2 \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang++ \
  -DCMAKE_CXX_FLAGS='-stdlib=libc++' \
  -DENABLE_TESTS=On \
  -DENABLE_BENCHMARKS=On \
  -DENABLE_GBENCHMARK=On \
  -DENABLE_WARNINGS_AS_ERRORS=Off \
  -DUMPIRE_ENABLE_BENCHMARKS=On \
  -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On \
  -DUMPIRE_ENABLE_DEVELOPER_BENCHMARKS=On \
  -DUMPIRE_ENABLE_CUDA=Off \
  -DUMPIRE_ENABLE_HIP=Off \
  -DUMPIRE_ENABLE_SYCL=Off \
  -DUMPIRE_ENABLE_OPENMP=Off \
  -DUMPIRE_ENABLE_OPENMP_TARGET=Off \
  -DCMAKE_BUILD_TYPE=Release
```

Host-only validation command:

```bash
cmake --build build-codex-2m2 \
  --target api_v2_host_memory_benchmarks api_v2_fixed_pool_benchmarks api_v2_allocator_benchmarks

build-codex-2m2/bin/api_v2_host_memory_benchmarks --benchmark_filter='BM_HostMemory_(Tracked|Untracked)/4096'
build-codex-2m2/bin/api_v2_fixed_pool_benchmarks --benchmark_filter='fixed_pool_(allocate|churn)'
build-codex-2m2/bin/api_v2_allocator_benchmarks --benchmark_filter='(VectorResize|MapInsert|Composition|Dispatch)'
```

Scope notes:

- This machine has no GPU or target-offload runtime, so device-backed
  benchmark execution is deferred to follow-up Beads work.
- The benchmark code remains backend-aware where possible, but only the host
  slice is validated here.
- Assembly-equivalence, binary-size, and compile-time measurements are tracked
  separately because they require dedicated inspection workflow rather than a
  benchmark executable alone.
