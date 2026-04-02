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

The current API v2 integration test executables are defined in
`tests/integration/CMakeLists.txt` and `tests/integration/api_v2/CMakeLists.txt`:

- `api_v2_registry_threading_tests`
- `api_v2_memory_threading_tests`
- `api_v2_thread_safe_stress_tests`
- `api_v2_stl_compatibility_tests`
- `api_v2_operations_headers_tests`
- `api_v2_operations_tests`

To run the full host-only API v2 test set after configuring and building:

```bash
ctest --test-dir build -R '^api_v2_' --output-on-failure
```

On the current `feature/api-refactor` branch, host validation should exclude two
pre-existing broken strategy tests that are tracked separately in Beads:

```bash
ctest --test-dir build \
  -R '^api_v2_' \
  -E 'api_v2_fixed_pool_tests|api_v2_dynamic_pool_list_tests' \
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
  -DUMPIRE_ENABLE_TESTS=On

cmake --build build --parallel
ctest --test-dir build \
  -R '^api_v2_' \
  -E 'api_v2_fixed_pool_tests|api_v2_dynamic_pool_list_tests' \
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
  -DUMPIRE_ENABLE_TESTS=On \
  -DUMPIRE_ENABLE_TOOLS=On \
  -DUMPIRE_ENABLE_ASAN=On \
  -DUMPIRE_ENABLE_SANITIZER_TESTS=On

cmake --build build-asan --parallel
ctest --test-dir build-asan \
  -R '^api_v2_' \
  -E 'api_v2_fixed_pool_tests|api_v2_dynamic_pool_list_tests' \
  --output-on-failure
```

ThreadSanitizer:

```bash
cmake -S . -B build-tsan \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang++ \
  -DCMAKE_C_FLAGS='-fsanitize=thread' \
  -DCMAKE_CXX_FLAGS='-stdlib=libc++ -fsanitize=thread' \
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

## Coverage

The repository already supports BLT coverage builds via `ENABLE_COVERAGE=On`.
The API v2-specific test wiring adds a `coverage_api_v2` target that filters the
coverage run to the `api_v2_*` CTest entries and builds the current host-safe
API v2 test executables before collecting coverage. The target currently
excludes `api_v2_fixed_pool_tests` and `api_v2_dynamic_pool_list_tests` until
their follow-up fixes land.

```bash
cmake -S . -B build-coverage \
  -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang++ \
  -DCMAKE_CXX_FLAGS='-stdlib=libc++' \
  -DUMPIRE_ENABLE_TESTS=On \
  -DENABLE_COVERAGE=On

cmake --build build-coverage --parallel
cmake --build build-coverage --target coverage_api_v2
```

When coverage tools are available, the generated report is written under
`build-coverage/coverage_api_v2/`.

## CI

The API v2 GitHub Actions workflow is defined in `.github/workflows/api_v2.yml`.
It runs host-only API v2 builds on Linux and macOS, adds sanitizer coverage for
the current API v2 targets, and uses the `coverage_api_v2` target for focused
coverage reporting.
