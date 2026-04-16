# API v2 Release Workload Suite

This directory contains the current in-repo benchmark sources used to define
the API v2 release-validation workload suite.

The goal of this suite is not to replace the broader API v2 unit and
integration tests. It defines the representative workloads and signoff
thresholds that downstream execution beads use before stable release.

## Suite Name

`api_v2_release_workload_suite_v1`

Until a checked-in downstream application fixture exists, release signoff uses
the following in-repo workload bundles as the representative workload suite.

## Workload Bundles

### 1. `host_interop_correctness`

Purpose:
- validate the host-side mixed v1/v2 ownership and operation paths that matter
  for migration and production rollout

Required coverage:
- `api_v2_v1_interop_tests`
- `api_v2_operations_tests`
- `api_v2_stl_compatibility_tests`

Pass criteria:
- all required executables return success
- no unexpected crashes, assertions, or sanitizer failures are permitted

### 2. `host_dynamic_pool_variable_size`

Purpose:
- exercise variable-size host allocation churn through
  `bench_dynamic_pool_list.cpp`

Representative patterns:
- random size allocations
- fragmentation patterns
- malloc comparison
- coalescing overhead
- variable-size stress
- pool growth

Pass criteria:
- benchmark completes all named phases without crash or allocator corruption
- median throughput and average latency do not regress by more than 15% versus
  the latest accepted baseline collected on the same machine, compiler, and
  build configuration

### 3. `host_quick_pool_small_object`

Purpose:
- exercise small-object host allocation churn through
  `bench_quick_pool.cpp`

Representative patterns:
- O(1) allocation verification
- quick_pool versus dynamic_pool_list
- quick_pool versus malloc
- size classes
- fragmentation
- multi-bin stress
- churn pattern
- large versus small
- scalability

Pass criteria:
- benchmark completes all named phases without crash or allocator corruption
- median throughput and average latency do not regress by more than 15% versus
  the latest accepted baseline collected on the same machine, compiler, and
  build configuration

### 4. `remote_device_workloads`

Purpose:
- validate the device and offload subset of the representative workload suite
  on hardware-capable systems

Applicable backends:
- CUDA
- HIP
- SYCL
- OpenMP target

Execution owner:
- `umpire-ds5`

Pass criteria:
- all applicable remote workloads complete without correctness failures
- backend-specific median throughput and average latency do not regress by more
  than 15% versus the latest accepted baseline on the same backend, compiler,
  driver, and runtime stack
- any skipped backend path or environment-specific failure must be captured as a
  bead before stable release

## Host-Safe Execution Path

The local host-safe execution bead is `umpire-v0s`.

Build the explicit API v2 host workload benchmark targets with the LLVM 19
host configuration:

```bash
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang++ \
  -DCMAKE_AR=/opt/homebrew/opt/llvm@19/bin/llvm-ar \
  -DCMAKE_RANLIB=/opt/homebrew/opt/llvm@19/bin/llvm-ranlib \
  -DCMAKE_CXX_FLAGS='-stdlib=libc++' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
  -DENABLE_BENCHMARKS=On \
  -DUMPIRE_ENABLE_TESTS=On \
  -DUMPIRE_ENABLE_BENCHMARKS=On

cmake --build build \
  --target api_v2_release_workload_host_benchmarks \
  --parallel
```

The checked-in benchmark targets are:
- `api_v2_dynamic_pool_list_benchmarks`
- `api_v2_quick_pool_benchmarks`
- `api_v2_release_workload_host_benchmarks` (aggregate target for the host-safe
  release workload benchmarks)

`UMPIRE_ENABLE_BENCHMARKS` is gated by BLT's top-level `ENABLE_BENCHMARKS`
option, so both options must be enabled for these targets to appear.

Run the host-safe benchmark bundle through the built benchmark binaries:

```bash
./build/bin/api_v2_dynamic_pool_list_benchmarks
./build/bin/api_v2_quick_pool_benchmarks
```

The execution bead must then:
- run the `host_interop_correctness` bundle from the built test targets
- execute the explicit benchmark binaries above using the same configured
  compiler and build settings as the host validation build
- record exact commands, raw results, and the baseline used for comparison

## Remote-Only Execution Path

The remote hardware-backed execution bead is `umpire-ds5`.

It must:
- reuse the suite definition and thresholds in this file
- run only on systems that provide the relevant device or offload backend
- record exact environment details alongside the benchmark and correctness
  results

## Threshold Rules

These rules apply to both host-safe and remote-only workload execution unless a
backend-specific bead documents a stricter requirement:

- correctness failures always fail release signoff
- throughput regressions greater than 15% fail release signoff
- average latency regressions greater than 15% fail release signoff
- missing baseline data does not fail the run, but the execution bead must
  publish the captured results as the candidate baseline and state that the run
  was baseline-establishing

## Reporting Requirements

Every workload-execution bead must record:
- the exact git commit under test
- compiler and build flags
- machine or runner identity
- the commands used for each workload bundle
- raw results and the derived pass or fail decision
- any follow-up bead created for regressions, skipped paths, or environment
  issues

## Follow-Up

- `umpire-z6w` tracks wiring these benchmark sources into explicit benchmark
  build targets or another audited execution path so workload execution does not
  depend on ad hoc compilation.
