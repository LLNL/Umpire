# Implementation Tasks

## 1. Foundation (Core Infrastructure)

- [ ] 1.1 Implement platform type tag structs (`host_platform`, `cuda_platform`, `hip_platform`, `sycl_platform`, `omp_target_platform`, `undefined_platform`) in `include/umpire/platform.hpp`
- [ ] 1.2 Implement `platform_for<Platform>` trait template mapping tags to `camp::resources::Platform` enum values
- [ ] 1.3 Add compile-time tests for platform trait mappings
- [ ] 1.4 Implement `allocation_record` struct with `ptr`, `size`, `strategy`, and optional `backtrace` in `include/umpire/allocation_record.hpp`
- [ ] 1.5 Add unit tests for `allocation_record` construction and field access

## 2. Registry Implementation

- [ ] 2.1 Implement `detail::registry` class skeleton in `include/umpire/detail/registry.hpp` and `src/umpire/detail/registry.cpp`
- [ ] 2.2 Add Meyer's Singleton pattern: `static registry& get()` method
- [ ] 2.3 Implement thread-safe unique ID generation via `std::atomic<int>` in `get_id()`
- [ ] 2.4 Add allocator registration: `allocator_list` (vector), `allocator_map` (by name), `allocator_id_map` (by ID)
- [ ] 2.5 Implement allocator lookup methods: `find_by_id()`, `find_by_name()`
- [ ] 2.6 Integrate with existing allocation map or create shared allocation map structure
- [ ] 2.7 Implement allocation tracking: `register_allocation(record)`, `find_allocation(ptr)`, `remove_allocation(ptr)`
- [ ] 2.8 Add mutex/lock for thread-safe map access (consider reader-writer lock for performance)
- [ ] 2.9 Add unit tests for registry singleton, ID generation uniqueness, allocator registration/lookup
- [ ] 2.10 Add thread safety tests: concurrent ID generation, concurrent registration, concurrent allocation tracking

## 3. Memory Base Class

- [ ] 3.1 Implement `memory` abstract base class in `include/umpire/memory.hpp` and `src/umpire/memory.cpp`
- [ ] 3.2 Add constructor accepting `std::string name`, auto-generating ID via `registry::get().get_id()`
- [ ] 3.3 Add pure virtual methods: `allocate(std::size_t)`, `deallocate(void*)`, `get_platform()`
- [ ] 3.4 Add non-virtual introspection: `get_name()`, `get_id()`, `get_current_size()`, `get_actual_size()`, `get_highwatermark()`
- [ ] 3.5 Add protected tracking methods: `track_allocation(void* ptr, std::size_t size)`, `untrack_allocation(void* ptr)`
- [ ] 3.6 Implement statistics tracking: `current_size_`, `actual_size_`, `highwatermark_` with atomic updates
- [ ] 3.7 Implement self-registration with registry in constructor
- [ ] 3.8 Add virtual destructor, deregister from registry on destruction
- [ ] 3.9 Add unit tests for base class introspection, tracking, statistics updates
- [ ] 3.10 Add tests for lifecycle: construction, registration, destruction, deregistration

## 4. Memory Resource Template

- [ ] 4.1 Implement `memory_resource<Platform>` template class inheriting from `memory` in `include/umpire/memory_resource.hpp`
- [ ] 4.2 Add `using platform = Platform` type alias for platform propagation
- [ ] 4.3 Add template parameters: `Allocator` (underlying allocator), `Tracking` (bool, default true)
- [ ] 4.4 Implement `get_platform()` override returning `platform_for<Platform>::value`
- [ ] 4.5 Add conditional tracking via `if constexpr(Tracking)` in derived allocate/deallocate
- [ ] 4.6 Add unit tests for platform type propagation and tracking toggle

## 5. Concrete Memory Resources

- [ ] 5.1 Implement `host_memory<Allocator = std::allocator<char>, Tracking = true>` in `include/umpire/resource/host_memory.hpp`
  - [ ] 5.1.1 Inherit from `memory_resource<host_platform>`
  - [ ] 5.1.2 Add Meyer's Singleton: `static host_memory& get()`
  - [ ] 5.1.3 Implement `allocate()` using `Allocator::allocate()` or `malloc()`
  - [ ] 5.1.4 Implement `deallocate()` using `Allocator::deallocate()` or `free()`
  - [ ] 5.1.5 Add unit tests: singleton access, allocation/deallocation, tracking on/off
- [ ] 5.2 Implement `cuda_device_memory<Allocator = cuda_allocator, Tracking = true>` in `include/umpire/resource/cuda_device_memory.hpp` (conditionally compiled if `UMPIRE_ENABLE_CUDA`)
  - [ ] 5.2.1 Inherit from `memory_resource<cuda_platform>`
  - [ ] 5.2.2 Add singleton pattern
  - [ ] 5.2.3 Implement `allocate()` using `cudaMalloc()`
  - [ ] 5.2.4 Implement `deallocate()` using `cudaFree()`
  - [ ] 5.2.5 Add error handling for CUDA API failures
  - [ ] 5.2.6 Add unit tests (requires CUDA runtime)
- [ ] 5.3 Implement `hip_device_memory<Allocator = hip_allocator, Tracking = true>` in `include/umpire/resource/hip_device_memory.hpp` (conditionally compiled if `UMPIRE_ENABLE_HIP`)
  - [ ] 5.3.1 Inherit from `memory_resource<hip_platform>`
  - [ ] 5.3.2 Add singleton pattern
  - [ ] 5.3.3 Implement `allocate()` using `hipMalloc()`
  - [ ] 5.3.4 Implement `deallocate()` using `hipFree()`
  - [ ] 5.3.5 Add error handling for HIP API failures
  - [ ] 5.3.6 Add unit tests (requires HIP runtime)
- [ ] 5.4 Implement `sycl_device_memory<Allocator = sycl_allocator, Tracking = true>` in `include/umpire/resource/sycl_device_memory.hpp` (conditionally compiled if `UMPIRE_ENABLE_SYCL`)
  - [ ] 5.4.1 Inherit from `memory_resource<sycl_platform>`
  - [ ] 5.4.2 Add singleton pattern
  - [ ] 5.4.3 Implement `allocate()` using SYCL USM allocation
  - [ ] 5.4.4 Implement `deallocate()` using SYCL free
  - [ ] 5.4.5 Add error handling for SYCL exceptions
  - [ ] 5.4.6 Add unit tests (requires SYCL runtime)
- [ ] 5.5 Implement `openmp_target_memory<Allocator = omp_allocator, Tracking = true>` in `include/umpire/resource/openmp_target_memory.hpp` (conditionally compiled if `UMPIRE_ENABLE_OPENMP`)
  - [ ] 5.5.1 Inherit from `memory_resource<omp_target_platform>`
  - [ ] 5.5.2 Add singleton pattern
  - [ ] 5.5.3 Implement `allocate()` using `omp_target_alloc()`
  - [ ] 5.5.4 Implement `deallocate()` using `omp_target_free()`
  - [ ] 5.5.5 Add unit tests (requires OpenMP target support)
- [ ] 5.6 Implement `null_resource` in `include/umpire/resource/null_resource.hpp`
  - [ ] 5.6.1 Inherit from `memory_resource<undefined_platform>`
  - [ ] 5.6.2 `allocate()` returns `nullptr` or throws `std::bad_alloc`
  - [ ] 5.6.3 `deallocate()` is no-op
  - [ ] 5.6.4 Document use case: testing, dry-runs, intentional failures
  - [ ] 5.6.5 Add unit tests

## 6. Allocation Strategy Base

- [ ] 6.1 Implement `allocation_strategy` abstract base class inheriting from `memory` in `include/umpire/strategy/allocation_strategy.hpp`
- [ ] 6.2 Add `memory* parent_` member for wrapped memory source
- [ ] 6.3 Add constructor accepting `memory* parent`
- [ ] 6.4 Add platform propagation from parent: `get_platform()` delegates to `parent_->get_platform()`
- [ ] 6.5 Define pure virtual `allocate()` and `deallocate()` for derived strategies to implement
- [ ] 6.6 Add unit tests for base strategy construction and platform propagation

## 7. Strategy: thread_safe

- [ ] 7.1 Implement `thread_safe<Memory>` template in `include/umpire/strategy/thread_safe.hpp`
- [ ] 7.2 Add `std::mutex mutex_` member
- [ ] 7.3 Wrap `allocate()` with `std::lock_guard<std::mutex>`
- [ ] 7.4 Wrap `deallocate()` with `std::lock_guard<std::mutex>`
- [ ] 7.5 Propagate platform type from `Memory::platform`
- [ ] 7.6 Add unit tests: single-threaded correctness, multi-threaded safety (no data races)
- [ ] 7.7 Add stress test: concurrent allocations/deallocations from multiple threads

## 8. Strategy: fixed_pool

- [ ] 8.1 Implement `fixed_pool<Memory>` template in `include/umpire/strategy/fixed_pool.hpp`
- [ ] 8.2 Add constructor accepting object size and objects-per-pool configuration
- [ ] 8.3 Pre-allocate pools from parent memory on construction
- [ ] 8.4 Implement `allocate()`: return from pool if available, allocate new pool if empty
- [ ] 8.5 Implement `deallocate()`: return object to free list
- [ ] 8.6 Implement `release()`: return unused pools to parent memory
- [ ] 8.7 Add unit tests: allocation from pool, pool growth, release, statistics
- [ ] 8.8 Add test for pool exhaustion and reallocation

## 9. Strategy: dynamic_pool_list

- [ ] 9.1 Implement `dynamic_pool_list<Memory>` template in `include/umpire/strategy/dynamic_pool_list.hpp`
- [ ] 9.2 Add configuration: initial pool size, growth factor
- [ ] 9.3 Maintain list of pools, allocate new pool when current exhausted
- [ ] 9.4 Implement `allocate()`: find free block in pools, grow if necessary
- [ ] 9.5 Implement `deallocate()`: mark block as free in appropriate pool
- [ ] 9.6 Implement `release()`: coalesce and return unused pools to parent
- [ ] 9.7 Add unit tests: dynamic growth, multi-pool allocation, coalescing, release

## 10. Strategy: quick_pool

- [ ] 10.1 Implement `quick_pool<Memory>` template in `include/umpire/strategy/quick_pool.hpp`
- [ ] 10.2 Use power-of-2 size bins for fast allocation (e.g., 16, 32, 64, 128, ..., 4096 bytes)
- [ ] 10.3 Maintain free list per bin
- [ ] 10.4 Implement `allocate()`: round up to next power-of-2, allocate from bin
- [ ] 10.5 Implement `deallocate()`: determine size bin, return to free list
- [ ] 10.6 Add configuration: bin sizes, blocks per bin
- [ ] 10.7 Add unit tests: power-of-2 allocation, bin selection, free list management
- [ ] 10.8 Add performance test comparing to direct allocation

## 11. Strategy: monotonic_buffer

- [ ] 11.1 Implement `monotonic_buffer<Memory>` template in `include/umpire/strategy/monotonic_buffer.hpp`
- [ ] 11.2 Allocate large buffer from parent on construction
- [ ] 11.3 Implement `allocate()`: bump pointer allocation (append-only)
- [ ] 11.4 Implement `deallocate()`: no-op (individual deallocations not supported)
- [ ] 11.5 Implement `release()`: reset pointer to beginning, optionally return buffer to parent
- [ ] 11.6 Add configuration: buffer size, growth strategy if exhausted
- [ ] 11.7 Add unit tests: bump allocation, no-op deallocation, release and reuse
- [ ] 11.8 Add test for buffer exhaustion behavior

## 12. Strategy: size_limiter

- [ ] 12.1 Implement `size_limiter<Memory>` template in `include/umpire/strategy/size_limiter.hpp`
- [ ] 12.2 Add constructor accepting size limit (bytes)
- [ ] 12.3 Track current allocation total
- [ ] 12.4 Implement `allocate()`: check if size would exceed limit, throw `umpire::logic_error` if exceeded, otherwise delegate to parent
- [ ] 12.5 Implement `deallocate()`: update current allocation total, delegate to parent
- [ ] 12.6 Add unit tests: allocation within limit, exception on exceeding limit, deallocation updates total
- [ ] 12.7 Add test for limit enforcement with multiple allocations

## 13. Strategy: named

- [ ] 13.1 Implement `named<Memory>` template in `include/umpire/strategy/named.hpp`
- [ ] 13.2 Add constructor accepting custom name string
- [ ] 13.3 Override `get_name()` to return custom name
- [ ] 13.4 Delegate `allocate()` and `deallocate()` to parent
- [ ] 13.5 Ensure allocations are tagged with custom name in registry
- [ ] 13.6 Add unit tests: name propagation, registry lookup by name, introspection
- [ ] 13.7 Add test for name uniqueness handling (if duplicate names allowed or not)

## 14. Typed Allocator

- [ ] 14.1 Implement `allocator<T, Memory>` template class in `include/umpire/allocator.hpp`
- [ ] 14.2 Add STL type aliases: `value_type`, `size_type`, `difference_type`, `pointer`, `const_pointer`, `reference`, `const_reference`
- [ ] 14.3 Add `using platform = typename Memory::platform` for platform propagation
- [ ] 14.4 Add constructor accepting `Memory*` pointer
- [ ] 14.5 Implement copy constructor and assignment operator
- [ ] 14.6 Implement equality comparison operators (`operator==`, `operator!=`)
- [ ] 14.7 Implement `allocate(size_type n)` returning `pointer` (type-safe `T*`)
  - [ ] 14.7.1 Multiply `n * sizeof(T)` to get byte count
  - [ ] 14.7.2 Call `memory_->allocate(bytes)`
  - [ ] 14.7.3 Cast `void*` to `T*` and return
- [ ] 14.8 Implement `deallocate(pointer ptr, size_type n)`
  - [ ] 14.8.1 Cast `T*` to `void*`
  - [ ] 14.8.2 Call `memory_->deallocate(ptr)` (size known from allocator, may not need to look up)
- [ ] 14.9 Add introspection methods delegating to `memory_`: `get_current_size()` (in units of T), `get_name()`, etc.
- [ ] 14.10 Implement `get_memory()` to access underlying `Memory*`
- [ ] 14.11 Add `rebind<U>` nested struct for STL compatibility
- [ ] 14.12 Add unit tests: construction, allocation returns correct type, deallocation, comparison
- [ ] 14.13 Add tests for platform type propagation and access

## 15. STL Compatibility

- [ ] 15.1 Add `using Allocator = allocator<char>` type alias for backward compatibility in `include/umpire/Allocator.hpp`
- [ ] 15.2 Test `std::vector<T, allocator<T, Memory>>` construction, resize, allocation
- [ ] 15.3 Test `std::map<K, V, std::less<K>, allocator<std::pair<const K, V>, Memory>>`
- [ ] 15.4 Test `std::unordered_map<K, V, ...>` with custom allocator
- [ ] 15.5 Test `std::allocate_shared<T>(alloc, args...)` with typed allocator
- [ ] 15.6 Test move semantics and container swap with custom allocators
- [ ] 15.7 Test allocator propagation on copy/move/swap per STL requirements
- [ ] 15.8 Add examples in documentation showing STL container usage

## 16. Memory Operations Integration

- [ ] 16.1 Implement `copy<SrcPlatform, DstPlatform>(void* dst, const void* src, std::size_t size)` template in `include/umpire/op/copy.hpp`
  - [ ] 16.1.1 Use `platform_for<SrcPlatform>` and `platform_for<DstPlatform>` to get enum values
  - [ ] 16.1.2 Look up operation: `MemoryOperationRegistry::getInstance().find("COPY", {src_plat, dst_plat})`
  - [ ] 16.1.3 Invoke operation: `op->transform(dst, src, size)`
  - [ ] 16.1.4 Handle case where operation not found (throw `umpire::runtime_error`)
- [ ] 16.2 Implement `memset<Platform>(void* ptr, int value, std::size_t size)` template in `include/umpire/op/memset.hpp`
  - [ ] 16.2.1 Use `platform_for<Platform>` to get enum
  - [ ] 16.2.2 Look up "MEMSET" operation
  - [ ] 16.2.3 Invoke operation
- [ ] 16.3 Implement `reallocate<Platform>(void* ptr, std::size_t size)` template in `include/umpire/op/reallocate.hpp`
  - [ ] 16.3.1 Look up "REALLOCATE" operation for platform
  - [ ] 16.3.2 Invoke and return new pointer
- [ ] 16.4 Implement `prefetch<Platform>(void* ptr, std::size_t size)` template (if applicable) in `include/umpire/op/prefetch.hpp`
- [ ] 16.5 Add error handling for unsupported platform pairs (throw with clear message)
- [ ] 16.6 Add unit tests for each operation template with supported platforms
- [ ] 16.7 Add tests for unsupported platform combinations (expect exception)
- [ ] 16.8 Add integration tests: allocate on platform A, copy to platform B, use on platform B

## 17. Error Handling

- [ ] 17.1 Define exception hierarchy in `include/umpire/error.hpp`:
  - [ ] 17.1.1 `umpire::out_of_memory` (inherits from `std::bad_alloc`)
  - [ ] 17.1.2 `umpire::unknown_allocation` (inherits from `std::runtime_error`)
  - [ ] 17.1.3 `umpire::runtime_error` (inherits from `std::runtime_error`)
  - [ ] 17.1.4 `umpire::logic_error` (inherits from `std::logic_error`)
- [ ] 17.2 Update allocate methods to throw `out_of_memory` on allocation failure
- [ ] 17.3 Update deallocate methods to throw `unknown_allocation` on untracked pointer (or assert in release mode)
- [ ] 17.4 Add tests for each exception type and condition
- [ ] 17.5 Document exception guarantees: allocate (strong), deallocate (no-throw for valid pointers)

## 18. Thread Safety Validation

- [ ] 18.1 Add thread safety tests for `detail::registry`: concurrent ID generation, concurrent registration
- [ ] 18.2 Add thread safety tests for `memory` base class: concurrent introspection (read-only)
- [ ] 18.3 Add thread safety tests for `thread_safe<>` wrapper: concurrent allocations, no data races
- [ ] 18.4 Add negative tests: concurrent allocation on non-thread-safe memory (expect data race in sanitizer builds)
- [ ] 18.5 Run all tests under ThreadSanitizer (TSan) to detect data races
- [ ] 18.6 Document thread safety guarantees in API docs for each class

## 19. Backward Compatibility & Interoperability

- [ ] 19.1 Ensure v2 allocations registered in shared allocation map visible to v1
- [ ] 19.2 Test v1 operations (copy, memset, etc.) on v2 allocations
- [ ] 19.3 Test v1 tools (replay, introspection) with v2 allocations
- [ ] 19.4 Add integration tests mixing v1 and v2 allocators in same application
- [ ] 19.5 Verify `Allocator` type alias resolves correctly for v1 code compatibility
- [ ] 19.6 Document compatibility matrix: v1 operations + v2 allocations, v2 operations + v1 allocations

## 20. Performance Benchmarks

- [ ] 20.1 Create benchmark suite in `tests/benchmarks/api_v2/`
- [ ] 20.2 Benchmark: host allocation with tracking on/off vs v1
- [ ] 20.3 Benchmark: GPU allocation with tracking on/off vs v1
- [ ] 20.4 Benchmark: strategy composition (1-3 layers) overhead
- [ ] 20.5 Benchmark: STL container operations (vector resize, map insert) with custom allocator vs std::allocator
- [ ] 20.6 Benchmark: platform dispatch via `if constexpr` (verify zero overhead)
- [ ] 20.7 Validate zero-cost abstractions via compiler explorer (assembly inspection)
- [ ] 20.8 Document performance results and identify any regressions vs v1

## 21. Documentation

- [ ] 21.1 Add Doxygen comments to all public classes, methods, and templates
- [ ] 21.2 Create user guide: `docs/sphinx/features/api_v2.rst`
  - [ ] 21.2.1 Introduction and motivation
  - [ ] 21.2.2 Core concepts: memory, allocator, strategies
  - [ ] 21.2.3 Platform types and compile-time dispatch
  - [ ] 21.2.4 Strategy composition patterns
  - [ ] 21.2.5 STL container integration
- [ ] 21.3 Create tutorial: `docs/tutorial/api_v2_tutorial.rst`
  - [ ] 21.3.1 Example 1: Basic host allocation with vector
  - [ ] 21.3.2 Example 2: GPU allocation with CUDA
  - [ ] 21.3.3 Example 3: Pooled allocation with fixed_pool
  - [ ] 21.3.4 Example 4: Thread-safe shared allocator
  - [ ] 21.3.5 Example 5: Strategy composition (thread_safe + pool)
  - [ ] 21.3.6 Example 6: Cross-platform memory copy
- [ ] 21.4 Create migration guide: `docs/sphinx/features/api_v2_migration.rst`
  - [ ] 21.4.1 V1 to V2 mappings (ResourceManager → registry, makeAllocator → direct construction)
  - [ ] 21.4.2 Common patterns translation
  - [ ] 21.4.3 Breaking changes (none, backward compatible)
  - [ ] 21.4.4 Deprecation timeline
- [ ] 21.5 Add code examples in `examples/api_v2/`
- [ ] 21.6 Update main README to mention v2 API
- [ ] 21.7 Create design rationale document (or use design.md from openspec)

## 22. Testing Infrastructure

- [ ] 22.1 Set up test directory structure: `tests/unit/api_v2/`, `tests/integration/api_v2/`
- [ ] 22.2 Configure CMake to build v2 tests
- [ ] 22.3 Add tests to CI pipeline (all platforms, compilers)
- [ ] 22.4 Add sanitizer builds (ASan, TSan, UBSan) for v2 tests
- [ ] 22.5 Add coverage reporting for v2 code
- [ ] 22.6 Document test running procedure

## 23. Build System Integration

- [ ] 23.1 Add CMake option `UMPIRE_ENABLE_API_V2` (default ON)
- [ ] 23.2 Add conditional compilation guards for platform-specific resources
- [ ] 23.3 Set up extern template instantiations for common types to reduce binary size
- [ ] 23.4 Update install targets to include v2 headers
- [ ] 23.5 Verify v2 headers are self-contained (can be included independently)
- [ ] 23.6 Test build on all supported platforms (Linux, macOS, Windows if applicable)

## 24. Code Review & Quality

- [ ] 24.1 Internal code review of all v2 components
- [ ] 24.2 Run static analysis (clang-tidy, cppcheck) on v2 code
- [ ] 24.3 Address warnings and potential issues
- [ ] 24.4 Verify consistent code style with Umpire conventions
- [ ] 24.5 Review error messages for clarity and usefulness
- [ ] 24.6 Review documentation for completeness and accuracy

## 25. Beta Preparation

- [ ] 25.1 Tag beta release in version control (e.g., v2024.08.0-beta)
- [ ] 25.2 Create beta announcement document
- [ ] 25.3 Identify beta test users/projects
- [ ] 25.4 Set up feedback mechanism (GitHub issues, discussion forum)
- [ ] 25.5 Create beta feedback questionnaire
- [ ] 25.6 Schedule beta review meetings

## 26. Final Release Preparation

- [ ] 26.1 Address all beta feedback and critical issues
- [ ] 26.2 Final documentation review and polish
- [ ] 26.3 Performance validation on representative workloads
- [ ] 26.4 Update changelog with v2 features
- [ ] 26.5 Prepare release notes
- [ ] 26.6 Tag stable release (e.g., v2025.01.0)
- [ ] 26.7 Publish user guide and tutorial
- [ ] 26.8 Announce release to community (blog post, mailing list, conferences)

---

## Implementation Order

Recommended order respecting dependencies:

1. **Phase 1: Foundation** (Tasks 1, 2, 3, 4, 6)
2. **Phase 2: Resources** (Task 5)
3. **Phase 3: Strategies** (Tasks 7-13)
4. **Phase 4: Allocator** (Tasks 14, 15)
5. **Phase 5: Operations** (Task 16)
6. **Phase 6: Quality** (Tasks 17, 18, 19, 20)
7. **Phase 7: Documentation** (Task 21)
8. **Phase 8: Integration** (Tasks 22, 23, 24)
9. **Phase 9: Release** (Tasks 25, 26)

## Estimated Effort

Rough estimates per phase (developer-weeks):

- Phase 1: 3-4 weeks
- Phase 2: 2-3 weeks
- Phase 3: 4-5 weeks
- Phase 4: 2-3 weeks
- Phase 5: 1-2 weeks
- Phase 6: 3-4 weeks
- Phase 7: 2-3 weeks
- Phase 8: 2-3 weeks
- Phase 9: 1-2 weeks

**Total: ~20-29 developer-weeks** (roughly 4-6 months for 1-2 developers)
