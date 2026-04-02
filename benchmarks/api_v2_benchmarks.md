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

Host-only validation command:

```bash
cmake --build build-codex-2m2 \
  --target api_v2_host_memory_benchmarks api_v2_fixed_pool_benchmarks api_v2_allocator_benchmarks

build-codex-2m2/bin/api_v2_host_memory_benchmarks --benchmark_filter='HostMemory_(Tracked|Untracked)$'
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
