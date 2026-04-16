# API v2 Host Release Workload Results

Bead: `umpire-v0s`

Commit under test: `fffef6b1`

Machine:
- Host-only macOS environment
- No CUDA, HIP, SYCL, or OpenMP target devices available locally

Build configuration:

```bash
cmake -S . -B build-v0s-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang++ \
  -DCMAKE_AR=/opt/homebrew/opt/llvm@19/bin/llvm-ar \
  -DCMAKE_RANLIB=/opt/homebrew/opt/llvm@19/bin/llvm-ranlib \
  -DCMAKE_CXX_FLAGS='-stdlib=libc++' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
  -DENABLE_TESTS=On \
  -DUMPIRE_ENABLE_TESTS=On \
  -DENABLE_BENCHMARKS=On \
  -DUMPIRE_ENABLE_BENCHMARKS=On

cmake --build build-v0s-release --parallel \
  --target api_v2_v1_interop_tests api_v2_operations_tests \
           api_v2_stl_compatibility_tests api_v2_release_workload_host_benchmarks
```

## Workload Bundle: `host_interop_correctness`

Command:

```bash
ctest --test-dir build-v0s-release \
  -R '^(api_v2_v1_interop_tests|api_v2_operations_tests|api_v2_stl_compatibility_tests)$' \
  --output-on-failure
```

Result:
- `api_v2_stl_compatibility_tests`: passed
- `api_v2_operations_tests`: passed
- `api_v2_v1_interop_tests`: passed
- Bundle status: pass

## Workload Bundle: `host_dynamic_pool_variable_size`

Command:

```bash
./build-v0s-release/bin/api_v2_dynamic_pool_list_benchmarks
```

Observed completion:
- Random pattern time: `25.625 ms`
- `malloc` comparison: `malloc 7.090 ms`, `dynamic_pool_list 8.621 ms`, speedup `0.82x`
- Variable size stress test: completed with `Allocated: 0 bytes`, `Block count: 1`
- Pool growth allocations: `0.55-0.56 ms`, deallocation `0.01-0.02 ms`
- Bundle status: pass for correctness/completion

## Workload Bundle: `host_quick_pool_small_object`

Command:

```bash
./build-v0s-release/bin/api_v2_quick_pool_benchmarks
```

Observed completion:
- Multi-bin stress test: `3.7 ms`, `53734551 ops/sec`
- Allocation churn pattern: `2 ms`, `102774923 ops/sec`
- Large vs small allocations: `1 ms` vs `6 ms`
- Scalability test completed through `10000 allocations`
- Bundle status: pass for correctness/completion

## Threshold Assessment

- Correctness thresholds: pass
- Performance regression thresholds: baseline-establishing run
- Release signoff decision for local host-safe execution: provisional pass

Notes:
- No accepted same-machine baseline was available in-repo for the benchmark bundles, so this run establishes the candidate baseline rather than evaluating the 15% regression threshold.
- Remote-only backend validation remains separate work in `umpire-ds5`.
