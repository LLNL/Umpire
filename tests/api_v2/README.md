# API v2 Testing

The API v2 tests reuse the repository's standard unit and integration test
layout. There is no separate API v2 test harness to configure.

## Test Targets

The current API v2 unit test executables are defined in
`tests/unit/CMakeLists.txt`:

- `api_v2_registry_tests`
- `api_v2_memory_tests`
- `api_v2_memory_resource_tests`
- `api_v2_allocator_tests`
- `api_v2_allocation_strategy_tests`
- `api_v2_host_memory_tests`
- `api_v2_null_resource_tests`
- `api_v2_thread_safe_tests`
- `api_v2_fixed_pool_tests`
- `api_v2_dynamic_pool_list_tests`
- `api_v2_quick_pool_tests`
- `api_v2_size_limiter_tests`
- `api_v2_monotonic_buffer_tests`
- `api_v2_named_tests`

Backend-specific unit tests are added when those backends are enabled:

- `api_v2_cuda_device_memory_tests`
- `api_v2_hip_device_memory_tests`
- `api_v2_sycl_device_memory_tests`
- `api_v2_openmp_target_memory_tests`

The current API v2 integration test executables are defined in
`tests/integration/CMakeLists.txt` and `tests/integration/api_v2/CMakeLists.txt`:

- `api_v2_registry_threading_tests`
- `api_v2_memory_threading_tests`
- `api_v2_thread_safe_stress_tests`
- `api_v2_stl_compatibility_tests`
- `api_v2_operations_headers_tests`
- `api_v2_operations_tests`
- `api_v2_v1_interop_tests`

To run the full host-only API v2 test set after configuring and building:

```bash
ctest --test-dir build \
  -R '^api_v2_' \
  --output-on-failure
```

## Local Build

On this repository, a host-only API v2 build is typically enough for day-to-day
development:

```bash
cmake -S . -B build \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang++ \
  -DCMAKE_CXX_FLAGS='-stdlib=libc++' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
  -DUMPIRE_ENABLE_TESTS=On

cmake --build build --parallel
ctest --test-dir build \
  -R '^api_v2_' \
  --output-on-failure
```

## Sanitizers

The repository already provides sanitizer options in
`cmake/SetupUmpireOptions.cmake` and sanitizer-specific tests in
`tests/tools/sanitizers/`.

AddressSanitizer:

```bash
cmake -S . -B build-asan \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang++ \
  -DCMAKE_CXX_FLAGS='-stdlib=libc++ -fsanitize=address' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
  -DUMPIRE_ENABLE_TESTS=On \
  -DUMPIRE_ENABLE_TOOLS=On \
  -DUMPIRE_ENABLE_ASAN=On \
  -DUMPIRE_ENABLE_SANITIZER_TESTS=On

cmake --build build-asan --parallel
ctest --test-dir build-asan \
  -R '^api_v2_' \
  --output-on-failure
```

ThreadSanitizer:

```bash
cmake -S . -B build-tsan \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang++ \
  -DCMAKE_C_FLAGS='-fsanitize=thread' \
  -DCMAKE_CXX_FLAGS='-stdlib=libc++ -fsanitize=thread' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
  -DUMPIRE_ENABLE_TESTS=On \
  -DUMPIRE_ENABLE_TSAN=On \
  -DUMPIRE_ENABLE_SANITIZER_TESTS=On

cmake --build build-tsan --parallel \
  --target api_v2_registry_threading_tests api_v2_memory_threading_tests api_v2_thread_safe_stress_tests
ctest --test-dir build-tsan \
  -R 'api_v2_registry_threading_tests|api_v2_memory_threading_tests|api_v2_thread_safe_stress_tests' \
  --output-on-failure
```

GPU-backed CUDA, HIP, SYCL, and OpenMP target test execution requires the
appropriate hardware and should be validated in dedicated follow-up tasks on
capable machines.

### HIP Validation

The API v2 HIP resource and operation paths should be validated on ROCm/HIP
systems rather than on the local macOS development machine. This repository
already carries the RADIUSS shared CI configuration under
``scripts/radiuss-spack-configs/``, including a HIP-focused job in
``scripts/radiuss-spack-configs/gitlab/radiuss-jobs/corona.yml`` named
``rocmcc_5_7_1_hip`` that builds a +rocm Umpire spec on the ``corona`` LC
system.

A representative manual validation recipe on a HIP-capable LC host for API v2
HIP device coverage is:

```bash
# On an LC system with ROCm/HIP (for example, corona with rocmcc HIP modules)
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_CXX_COMPILER=/opt/rocm-6.4.3/bin/amdclang++ \
  -DROCM_PATH=/opt/rocm-6.4.3 \
  -DENABLE_HIP=On \
  -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On \
  -DUMPIRE_ENABLE_TESTS=On

cmake --build build-hip --parallel \
  --target api_v2_hip_device_memory_tests api_v2_operations_tests

ctest --test-dir build-hip \
  -R '^(api_v2_hip_device_memory_tests|api_v2_operations_tests)$' \
  --output-on-failure
```

This flow does not run on the current macOS development host but documents a
concrete HIP-capable environment and configure/test invocation that follow-up
beads such as ``umpire-4og`` can use when exercising API v2 HIP device
coverage on ROCm-capable hardware.

### OpenMP Target Validation

The API v2 OpenMP target resource and copy paths are intended to be validated
on LC systems that provide an OpenMP target offload runtime rather than on the
local macOS development machine. This repository already carries the
RADIUSS shared CI configuration under
``scripts/radiuss-spack-configs/``, including machine pipelines in
``scripts/radiuss-spack-configs/gitlab/radiuss-jobs/tioga.yml`` and an
Umpire Spack package with an ``+omptarget`` variant that sets
``UMPIRE_ENABLE_OPENMP_TARGET`` in the generated host-config.

A representative manual validation recipe on an OpenMP target-capable LC host
for API v2 operations is:

```bash
# On an LC system with OpenMP target support (for example, tioga with CCE)
cmake -S . -B build-omptarget -G Ninja \
  -DCMAKE_C_COMPILER=cc \
  -DCMAKE_CXX_COMPILER=CC \
  -DENABLE_OPENMP=On \
  -DUMPIRE_ENABLE_OPENMP_TARGET=On \
  -DUMPIRE_ENABLE_TESTS=On

cmake --build build-omptarget --parallel \
  --target api_v2_openmp_target_memory_tests api_v2_operations_tests

ctest --test-dir build-omptarget \
  -R '^(api_v2_openmp_target_memory_tests|api_v2_operations_tests)$' \
  --output-on-failure
```

This flow does not run on the current macOS development host but documents a
concrete OpenMP target-capable environment and configure/test invocation that
downstream validation work can use when exercising ``openmp_target_memory`` on
target-enabled hardware. The
``api_v2_openmp_target_memory_tests`` target covers resource construction,
tracking, allocation, host-to-target copies, and basic deallocation semantics,
but actual execution still requires an OpenMP target-capable runtime.

## Static Analysis

On macOS with Homebrew LLVM 19, `clang-tidy` needs the active SDK sysroot to
find libc++ and platform headers correctly. The configured build directory must
also contain `compile_commands.json`, so the examples above enable
`-DCMAKE_EXPORT_COMPILE_COMMANDS=On`:

```bash
SDKROOT=$(xcrun --show-sdk-path)

/opt/homebrew/opt/llvm@19/bin/clang-tidy \
  -p build \
  --extra-arg=-isysroot \
  --extra-arg="$SDKROOT" \
  tests/integration/api_v2/test_v1_v2_interop.cpp \
  src/umpire/api_v2_instantiations.cpp
```

The representative `cppcheck` pass used for API v2 host validation should
enable inline suppressions and explicitly suppress the known
`noExplicitConstructor` false positives on the intentionally implicit
`allocator<T, Memory>` rebind constructor:

```bash
cppcheck \
  --inline-suppr \
  --suppress=noExplicitConstructor:include/umpire/allocator.hpp \
  --suppress=noExplicitConstructor:include/umpire/Allocator.hpp \
  --enable=warning,style,performance,portability \
  --std=c++17 \
  --language=c++ \
  --quiet \
  --error-exitcode=1 \
  -I include \
  -I src \
  tests/integration/api_v2/test_v1_v2_interop.cpp \
  src/umpire/api_v2_instantiations.cpp
```

## Device Validation CI Path

This repository cannot run CUDA, HIP, SYCL, or OpenMP target tests on the
current macOS development machine, but the CI configuration already includes
GPU-capable runners and Docker targets that future device-validation beads can
use.

- The shared Dockerfile defines device-oriented build targets:
  - `cuda`, `cuda13` (CUDA-capable images)
  - `hip` (HIP-capable image)
  - `sycl` (Intel oneAPI/SYCL-capable image)
- The `.github/workflows/build.yml` workflow uses these targets on:
  - `radiuss-cpu-runners` for generic builds (gcc/clang/tsan/hip/sycl/intel)
  - `radiuss-cuda-runners` for the `build_gpu` CUDA matrix
- The API v2 workflow `.github/workflows/api_v2.yml` exposes manual backend
  validation jobs:
  - `device_cuda_validate` builds CUDA validation images
    (`api_v2_cuda_validate`, `api_v2_cuda13_validate`) on
    `radiuss-cuda-runners` and runs
    `api_v2_cuda_device_memory_tests` plus `api_v2_operations_tests`
    inside those containers.
  - `device_sycl_validate` builds `api_v2_sycl_validate` on
    `radiuss-cpu-runners` and runs
    `api_v2_sycl_device_memory_tests` and `api_v2_operations_tests` in
    that container.

An operator with access to the GitHub-hosted repository and Actions can run the
CUDA and SYCL device workflows as follows:

1. Open the repository's "Actions" tab and select the "API v2 Tests" workflow.
2. Use "Run workflow" (the manual `workflow_dispatch` entry point) on the
   desired branch (for example, `feature/api-refactor`).
3. Wait for the workflow to start and inspect the `device_cuda_validate` and
   `device_sycl_validate` jobs to see pass/skip/failure output for the CUDA and
   SYCL API v2 device tests.

This wiring intentionally does not claim device validation is complete; it
only guarantees that a hardware-backed CI path exists for future API v2
device-validation work once suitable hosts are provisioned.

## Coverage

The repository already supports BLT coverage builds via `ENABLE_COVERAGE=On`.
The API v2-specific test wiring adds a `coverage_api_v2` target that filters the
coverage run to the `api_v2_*` CTest entries and builds the current host-safe
API v2 test executables, including `api_v2_fixed_pool_tests`, before collecting
coverage.

```bash
cmake -S . -B build-coverage \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang++ \
  -DCMAKE_CXX_FLAGS='-stdlib=libc++' \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=On \
  -DUMPIRE_ENABLE_TESTS=On \
  -DENABLE_COVERAGE=On

cmake --build build-coverage --parallel
cmake --build build-coverage --target coverage_api_v2
```

When coverage tools are available, the generated report is written under
`build-coverage/coverage_api_v2/`.

## Host Compatibility Matrix

The current host-only interoperability coverage validates these supported
combinations:

| Scenario | Status | Coverage |
|----------|--------|----------|
| v2 `host_memory<>::get()` allocation visible to v1 `ResourceManager` | Supported | `api_v2_v1_interop_tests` |
| v1 `ResourceManager::memset()` on v2 host allocation | Supported | `api_v2_v1_interop_tests` |
| v1 `ResourceManager::copy()` between v1/v2 host allocations | Supported | `api_v2_v1_interop_tests` |
| v1 `ResourceManager::deallocate()` on v2 host allocation | Supported | `api_v2_v1_interop_tests` |
| v1 `ResourceManager::reallocate(ptr, 0)` on v2 host allocation | Supported | `api_v2_v1_interop_tests` |
| v1 `ResourceManager::reallocate(ptr, 0, ctx)` on v2 host allocation | Supported | `api_v2_v1_interop_tests` |
| v1 `ResourceManager::move(ptr, HOST)` on v2 host allocation | Supported | `api_v2_v1_interop_tests` |
| v1 `ResourceManager::move(ptr, distinct host allocator)` on v2 host allocation | Supported | `api_v2_v1_interop_tests` |
| v1 `ResourceManager::reallocate(ptr, size, HOST)` on v2 host allocation | Supported | `api_v2_v1_interop_tests` |
| v1 `ResourceManager::reallocate(ptr, size, distinct host allocator)` on v2 host allocation | Rejected with `umpire::runtime_error` | `api_v2_v1_interop_tests` |

The host compatibility table above is intentionally narrower than the broader
legacy-surface migration audit in
`docs/sphinx/features/api_v2_migration.rst`. Use this table for checked host
interop behavior and the migration guide for the implementation-facing
classification of the relevant v1 `ResourceManager` and `Allocator` entry
points.

## CI

The API v2 GitHub Actions workflow is defined in `.github/workflows/api_v2.yml`.
It runs host-only API v2 builds on Linux and macOS, adds sanitizer coverage for
the current API v2 targets, and uses the `coverage_api_v2` target for focused
coverage reporting.
