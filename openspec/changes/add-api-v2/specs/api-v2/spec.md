# Umpire API v2.0 Specification

## Purpose

This specification defines the next-generation Umpire API, focusing on three major architectural changes:

1. **New Registry-Centered API** - New API entry points use a lightweight registry pattern instead of the monolithic `ResourceManager` singleton
2. **Typed Allocator Class** - Template-based allocator with STL compatibility
3. **Memory Concept** - Unified abstract base class hierarchy for all memory sources

### Design Goals

- Zero-cost abstractions via template-based design
- Compile-time platform information through type tags
- Optional tracking overhead configurable at compile time
- STL compatibility for seamless container integration
- Composable strategies using the decorator pattern
- Thread-safe by design with opt-in synchronization

---

## ADDED Requirements

### Requirement: Memory Base Class

The system SHALL provide an abstract base class `memory` that serves as the foundation for all memory sources in Umpire, replacing direct ResourceManager interaction with self-registration and unified tracking.

The `memory` class SHALL:
- Accept a name string during construction
- Provide pure virtual `allocate(std::size_t n)` and `deallocate(void* ptr)` methods
- Provide pure virtual `get_platform()` method returning `camp::resources::Platform`
- Provide non-virtual introspection methods: `get_current_size()`, `get_actual_size()`, `get_highwatermark()`, `get_name()`, `get_id()`
- Self-register with the `detail::registry` singleton upon construction
- Provide protected `track_allocation` and `untrack_allocation` methods for derived classes
- Maintain statistics: `current_size_`, `actual_size_`, and `highwatermark_`

#### Scenario: Memory instance construction and registration
- **WHEN** a new `memory` instance is constructed with a name
- **THEN** the instance obtains a unique ID from the registry
- **AND** the instance is registered in the registry's allocator maps

#### Scenario: Allocation tracking
- **WHEN** a derived class calls `track_allocation` after allocating memory
- **THEN** the allocation is recorded with pointer, size, and owning strategy
- **AND** statistics (`current_size_`, `highwatermark_`) are updated

#### Scenario: Introspection
- **WHEN** `get_current_size()` is called on a memory instance
- **THEN** the current allocated size in bytes is returned without virtual dispatch

---

### Requirement: Memory Hierarchy

The system SHALL provide a class hierarchy where:
- `memory` is the abstract base
- `memory_resource<Platform>` is a template for concrete memory sources
- `allocation_strategy` is an abstract base for decorators/pools

The hierarchy SHALL support:
- Platform-specific resources: `host_memory`, `cuda_device_memory`, `hip_device_memory`, `sycl_device_memory`, `openmp_target_memory`, `null_resource`
- Strategy wrappers: `fixed_pool`, `dynamic_pool_list`, `quick_pool`, `monotonic_buffer`, `thread_safe`, `size_limiter`, `named`

#### Scenario: Platform-specific memory resource
- **WHEN** a `cuda_device_memory` instance is created
- **THEN** it inherits from `memory_resource<cuda_platform>`
- **AND** its `platform` type alias resolves to `cuda_platform`

#### Scenario: Strategy composition
- **WHEN** a `thread_safe<fixed_pool<host_memory<>>>` is instantiated
- **THEN** allocations are mutex-protected
- **AND** underlying allocations use the fixed pool backed by host memory

---

### Requirement: Registry Pattern

The system SHALL provide a `detail::registry` class that serves as the central registry for allocator identity and allocation tracking in the new API, while coexisting with the existing `ResourceManager` and `MemoryOperationRegistry`.

The registry SHALL:
- Use Meyer's Singleton pattern for access via `registry::get()`
- Provide thread-safe unique ID generation via `get_id()`
- Maintain `allocator_list`, `allocator_map` (by name), and `allocator_id_map` (by ID)
- Maintain or wrap a global `allocation_map` for pointer-to-record lookup that is visible to existing tooling and operations
- Be non-copyable

The registry SHALL NOT:
- Create or manage allocator lifecycles
- Provide allocator factory methods
- Store default allocator state
- Implement memory operations

New API components SHALL obtain allocator identity and allocation records through `detail::registry` instead of `ResourceManager`. Existing APIs MAY continue to use `ResourceManager` as a façade over the same underlying allocation maps for backward compatibility.

#### Scenario: Singleton access
- **WHEN** `registry::get()` is called from multiple threads
- **THEN** all calls return the same registry instance
- **AND** no data races occur

#### Scenario: Unique ID generation
- **WHEN** `get_id()` is called
- **THEN** a unique integer ID is returned
- **AND** concurrent calls never return the same ID

#### Scenario: Allocation lookup
- **WHEN** a pointer is passed to find its owning allocator
- **THEN** the registry's allocation map provides O(1) lookup to the allocation record

---

### Requirement: Typed Allocator

The system SHALL provide a template class `allocator<T, Memory>` that provides type-safe memory allocation with full STL Allocator concept compliance.

The allocator SHALL define:
- STL type aliases: `size_type`, `difference_type`, `pointer`, `const_pointer`, `reference`, `const_reference`, `value_type`
- Platform type propagation via `using platform = typename Memory::platform`
- Constructor accepting `Memory*`
- Copy constructor and assignment
- Equality comparison operators
- `allocate(size_type n)` returning `pointer`
- `deallocate(pointer ptr, size_type n)` where `n` is the count of `T` objects (NOT bytes)
- Introspection methods returning counts in units of T
- `get_memory()` to access the underlying memory source

The system SHALL provide `using Allocator = allocator<char>` for backward compatibility.

#### Scenario: Basic typed allocation
- **WHEN** `allocator<double>` is constructed with a host memory resource
- **AND** `allocate(100)` is called
- **THEN** 100 doubles worth of memory (800 bytes) is allocated
- **AND** a `double*` pointer is returned

#### Scenario: STL container integration
- **WHEN** `std::vector<int, umpire::allocator<int, cuda_device_memory<>>>` is created
- **AND** `resize(1000)` is called
- **THEN** memory is allocated on the GPU device

#### Scenario: Platform-aware code
- **WHEN** template code inspects `Alloc::platform`
- **THEN** compile-time platform detection enables specialized implementations via `if constexpr`

#### Scenario: Deallocation with size
- **WHEN** `deallocate(ptr, 100)` is called on `allocator<double>`
- **THEN** the allocator knows to deallocate 800 bytes (100 * sizeof(double))
- **AND** no registry lookup is required for size

#### Scenario: Deallocation size consistency
- **WHEN** `deallocate(ptr, n)` is called
- **AND** `n` does not match the original `allocate(n)` count
- **THEN** behavior is undefined (programming error)
- **AND** this is documented as a precondition violation

---

### Requirement: Platform Types

The system SHALL provide platform tag types for compile-time platform identification:
- `undefined_platform`
- `host_platform`
- `cuda_platform`
- `hip_platform`
- `omp_target_platform`
- `sycl_platform`

The system SHALL provide a `platform_for<Platform>` trait mapping tags to `camp::resources::Platform` values.

#### Scenario: Platform tag usage
- **WHEN** code needs to specialize behavior for CUDA
- **THEN** `std::is_same_v<platform, cuda_platform>` can be used in `if constexpr`

#### Scenario: Platform trait consistency
- **WHEN** `platform_for<cuda_platform>::value` is accessed
- **THEN** it equals `camp::resources::Platform::cuda`
- **AND** this mapping is consistent with existing Umpire platform enums

---

### Requirement: Memory Resource Template

The system SHALL provide a `memory_resource<Platform>` template class inheriting from `memory` that concrete memory sources derive from.

Each concrete resource SHALL:
- Be a template with `Allocator` and `Tracking` parameters
- Provide singleton access via static `get()` method
- Implement `allocate`, `deallocate`, and `get_platform`
- Use `if constexpr(Tracking)` for zero-cost optional tracking

#### Scenario: Host memory singleton
- **WHEN** `host_memory<>::get()` is called
- **THEN** the singleton host memory resource is returned

#### Scenario: Tracking toggle
- **WHEN** `host_memory<malloc_allocator, false>` is used
- **THEN** allocations bypass tracking for zero overhead
- **WHEN** `host_memory<malloc_allocator, true>` is used
- **THEN** allocations are tracked via `track_allocation`

---

### Requirement: Allocation Strategies

The system SHALL provide allocation strategy classes that wrap memory resources to provide pooling, thread-safety, limits, and other functionality.

**Fixed Pool** SHALL:
- Accept object size and objects-per-pool configuration
- Provide `release()` to return unused pools to parent

**Thread-Safe Wrapper** SHALL:
- Add mutex protection to any memory source
- Propagate platform type from wrapped memory

**Size Limiter** SHALL:
- Enforce allocation quotas
- Throw if limit exceeded during `allocate`

#### Scenario: Fixed pool allocation
- **WHEN** `fixed_pool` is created with 64-byte objects and 1024 objects per pool
- **AND** `allocate(64)` is called
- **THEN** memory is returned from the pre-allocated pool

#### Scenario: Thread-safe allocation
- **WHEN** `thread_safe<pool>` is used from multiple threads
- **THEN** all allocations and deallocations are serialized via mutex

#### Scenario: Size limit enforcement
- **WHEN** `size_limiter` is configured with 1MB limit
- **AND** allocations exceed 1MB total
- **THEN** an exception is thrown

---

### Requirement: Memory Operations Integration

The system SHALL integrate with Umpire's existing memory operations (`copy`, `memset`, `reallocate`, `prefetch`) by providing template-based front-end APIs that dispatch to the existing `MemoryOperationRegistry` and concrete `MemoryOperation` implementations based on platform types.

The integration layer SHALL:
- Provide function templates (and, where appropriate, thin template wrappers) parameterized by platform tag types
- Use `platform_for<Platform>` to map platform tags to `camp::resources::Platform` values
- Use the existing `MemoryOperationRegistry` to look up the correct `MemoryOperation` by name and pair of `Platform` enums
- Support cross-platform operations (e.g., host-to-device copy) using the pre-existing operation implementations
- Throw `umpire::runtime_error` with a clear message when runtime platform lookup reaches a platform pair that the current build does not dispatch

The integration layer SHALL NOT:
- Change or replace the existing `MemoryOperation` subclasses
- Change or replace the `MemoryOperationRegistry` lookup behavior

Direct template wrappers SHALL only be available for the platform tags and platform pairs whose `op::*` specializations are compiled into the current build. Disabled backends and unsupported direct template pairs SHALL therefore be rejected at compile time rather than deferred to a runtime error path.

#### Scenario: Platform-dispatched copy
- **WHEN** a new API function template `copy<SrcPlatform, DstPlatform>(void* dst, const void* src, std::size_t size)` is called
- **THEN** it uses `platform_for<SrcPlatform>` and `platform_for<DstPlatform>` to obtain `camp::resources::Platform` values
- **AND** it uses `MemoryOperationRegistry::getInstance().find("COPY", {src_platform, dst_platform})` to obtain the operation
- **AND** it invokes the returned `MemoryOperation`'s `transform` or `transform_async` method

#### Scenario: Same-platform operation
- **WHEN** a new API function template `memset<Platform>(void* ptr, int value, std::size_t size)` is called
- **THEN** it uses `platform_for<Platform>` to obtain the `camp::resources::Platform` value
- **AND** it uses `MemoryOperationRegistry::getInstance().find("MEMSET", {platform, platform})` to obtain the operation
- **AND** the platform-specific memset implementation is used

#### Scenario: Unsupported runtime platform pair
- **WHEN** `copy<sycl_platform, cuda_platform>()` is called
- **AND** runtime platform dispatch reaches a platform combination that the current build does not support
- **THEN** `umpire::runtime_error` is thrown
- **AND** the error message indicates the unsupported platform pair

#### Scenario: Unsupported direct template pair is unavailable
- **WHEN** code attempts to instantiate a direct template wrapper for a disabled backend or for a platform pair with no compiled `op::*` specialization
- **THEN** the program is ill-formed at compile time
- **AND** the call does not reach runtime dispatch

#### Scenario: Operation error propagation
- **WHEN** a platform-specific operation (e.g., `cudaMemcpy`) fails
- **THEN** the template wrapper catches the error
- **AND** throws `umpire::runtime_error` with the underlying error details

---

### Requirement: Allocation Record

The system SHALL provide an `allocation_record` struct to track metadata for each allocation.

The record SHALL contain:
- `void* ptr` - allocation address
- `std::size_t size` - allocation size in bytes
- `memory* strategy` - owning memory source
- Optional `util::backtrace allocation_backtrace` when `UMPIRE_ENABLE_BACKTRACE` is defined

#### Scenario: Allocation record creation
- **WHEN** `track_allocation` is called
- **THEN** an `allocation_record` is created and stored in the registry's allocation map

#### Scenario: Allocation lookup for deallocation
- **WHEN** `deallocate(ptr)` is called with only a pointer
- **THEN** the allocation record is retrieved to determine size and owning strategy

---

### Requirement: Backward Compatibility

The system SHALL maintain backward compatibility with existing Umpire code through:
- `Allocator` type alias for `allocator<char>`
- Existing operations in `umpire/op/` continuing to work unchanged
- Allocations registered with new API visible in global allocation map
- Thread safety as opt-in via `thread_safe<>` wrapper

#### Scenario: Legacy code compatibility
- **WHEN** existing code uses `umpire::Allocator`
- **THEN** it compiles and functions as before via the type alias

#### Scenario: Debugging tool compatibility
- **WHEN** allocations are made with the new API
- **THEN** they appear in the global allocation map for debugging and replay tools

#### Scenario: Mixed API allocations visible together
- **WHEN** allocations are made with both v1 `ResourceManager` and v2 `allocator<>`
- **THEN** both appear in the global allocation map
- **AND** debugging tools (replay, introspection) see all allocations

#### Scenario: V1 operations work with V2 allocations
- **WHEN** a v2 `allocator<T, host_memory>` allocates memory
- **AND** v1 `copy()` operation is used on that pointer
- **THEN** the operation succeeds
- **AND** the registry provides allocation metadata to v1 operations

---

### Requirement: Error Handling

The system SHALL provide consistent error handling across all memory operations.

The error handling SHALL:
- Throw `umpire::out_of_memory` (derived from `std::bad_alloc`) when allocation fails
- Throw `umpire::unknown_allocation` when deallocating untracked pointers
- Throw `umpire::runtime_error` for platform-specific failures (e.g., cudaMalloc failure)
- Throw `umpire::logic_error` for invalid operations (e.g., size limit exceeded)
- Provide strong exception safety for allocate (no side effects if allocation fails)
- Provide no-throw guarantee for deallocate with valid pointers

#### Scenario: Out of memory
- **WHEN** `allocate()` is called but underlying platform allocation fails
- **THEN** `umpire::out_of_memory` is thrown
- **AND** no partial allocation is recorded in the registry

#### Scenario: Invalid deallocation
- **WHEN** `deallocate()` is called with a pointer not tracked by the registry
- **THEN** `umpire::unknown_allocation` is thrown
- **AND** the error message includes the pointer address

#### Scenario: Size limit exceeded
- **WHEN** `size_limiter` is used and allocation would exceed configured limit
- **THEN** `umpire::logic_error` is thrown with descriptive message
- **AND** current size statistics are not modified

#### Scenario: Platform-specific failure
- **WHEN** a CUDA allocation fails due to device error
- **THEN** `umpire::runtime_error` is thrown
- **AND** the error message includes the underlying CUDA error code

---

### Requirement: Thread Safety

The system SHALL provide thread safety guarantees for concurrent operations on memory instances.

The base `memory` class SHALL:
- Support concurrent read-only introspection methods (`get_name()`, `get_id()`, `get_platform()`) without external synchronization
- Require external synchronization or `thread_safe<>` wrapper for concurrent `allocate()` and `deallocate()` on the same instance
- Use atomic operations for statistics updates (`current_size_`, `highwatermark_`)
- Allow concurrent allocations on different memory instances without synchronization

The `detail::registry` SHALL:
- Provide thread-safe ID generation via atomic increment
- Protect allocator maps with appropriate synchronization (mutex or reader-writer lock)
- Support concurrent registration and lookup operations without data races
- Use thread-safe data structures or synchronization for allocation map access

#### Scenario: Concurrent introspection
- **WHEN** multiple threads call `get_current_size()` on the same memory instance
- **THEN** all threads receive consistent values without locking
- **AND** no data races occur

#### Scenario: Concurrent allocation without wrapper
- **WHEN** two threads call `allocate()` on the same `host_memory` instance without `thread_safe<>` wrapper
- **THEN** behavior is undefined (data race)
- **AND** this is documented as requiring user synchronization

#### Scenario: Concurrent allocation with wrapper
- **WHEN** two threads call `allocate()` on `thread_safe<host_memory>` instance
- **THEN** allocations are serialized via mutex
- **AND** both allocations succeed independently

#### Scenario: Concurrent registry access
- **WHEN** one thread registers a new allocator while another looks up an existing one
- **THEN** both operations complete without data races
- **AND** the registry remains in a valid state

---

### Requirement: Lifecycle Management

The system SHALL define lifecycle rules for memory instances and their allocations.

Memory instances SHALL:
- Remain valid for the lifetime of the process (singletons) or until explicitly destroyed
- Not automatically deallocate tracked allocations on destruction
- Emit a warning (via `UMPIRE_LOG`) if destroyed with active allocations
- Remove themselves from the registry on destruction

The system SHALL:
- Allow manual destruction of non-singleton memory instances
- Support detection of memory leaks via registry queries
- Provide undefined behavior if allocations outlive their memory source

#### Scenario: Singleton lifecycle
- **WHEN** `host_memory::get()` is called
- **THEN** the singleton is created on first access
- **AND** it lives until program termination

#### Scenario: Memory instance destroyed with active allocations
- **WHEN** a non-singleton memory instance is destroyed
- **AND** it has active tracked allocations
- **THEN** a warning is logged to `UMPIRE_LOG`
- **AND** the allocations are NOT automatically deallocated
- **AND** the memory instance is removed from registry

#### Scenario: Leak detection
- **WHEN** `registry::get().find_allocations_by_memory(memory_id)` is called before destruction
- **THEN** all active allocations for that memory instance are returned
- **AND** leak detection tools can enumerate them

---

### Requirement: Strategy Specifications

The system SHALL provide complete specifications for all allocation strategy wrappers.

**Dynamic Pool List** SHALL:
- Maintain a growable list of memory pools
- Allocate new pools when existing pools are exhausted
- Support configurable initial pool size and growth factor
- Provide `release()` to coalesce and return unused pools to parent

**Quick Pool** SHALL:
- Use power-of-2 size bins for fast allocation
- Maintain per-bin free lists
- Round allocation sizes up to next power-of-2
- Support configurable bin sizes and blocks per bin

**Monotonic Buffer** SHALL:
- Allocate from a large buffer using bump-pointer allocation
- Treat individual `deallocate()` calls as no-ops
- Provide `release()` to reset buffer and optionally return to parent
- Support configurable buffer size and growth strategy

**Named Strategy** SHALL:
- Accept a custom name string in constructor
- Override `get_name()` to return the custom name
- Delegate allocations to parent memory source
- Tag allocations with custom name in registry for debugging

**Null Resource** SHALL:
- Inherit from `memory_resource<undefined_platform>`
- Have `allocate()` throw `umpire::out_of_memory`
- Have `deallocate()` perform no operation
- Be useful for testing and dry-run scenarios

#### Scenario: Dynamic pool list growth
- **WHEN** `dynamic_pool_list<host_memory>` runs out of pre-allocated blocks
- **THEN** a new pool is allocated from the parent
- **AND** the pool list grows dynamically
- **AND** subsequent allocations use the new pool

#### Scenario: Quick pool fast path
- **WHEN** `quick_pool<host_memory>` is used with power-of-two sizes
- **THEN** allocations use fast bin lookup without search
- **AND** deallocations return to per-size free lists

#### Scenario: Monotonic buffer allocation
- **WHEN** `monotonic_buffer<host_memory>` allocates memory
- **THEN** the buffer pointer is incremented (bump allocation)
- **AND** individual deallocations are no-ops
- **WHEN** `release()` is called
- **THEN** the buffer pointer resets to the beginning

#### Scenario: Named strategy usage
- **WHEN** `named<host_memory>` is created with name "my-allocator"
- **THEN** the name appears in registry lookups
- **AND** `get_name()` returns "my-allocator"
- **AND** allocations are tagged with this name for debugging

#### Scenario: Null resource behavior
- **WHEN** `null_resource::allocate()` is called
- **THEN** `umpire::out_of_memory` is thrown
- **WHEN** `null_resource::deallocate()` is called
- **THEN** no operation is performed

## Design Patterns Summary

| Pattern | Usage | Location |
|---------|-------|----------|
| Meyer's Singleton | Registry and per-resource instances | `registry::get()`, `host_memory::get()` |
| Template Strategy | Pluggable allocation strategies | `memory`, strategies |
| Decorator | Wrapper strategies | `thread_safe`, `size_limiter` |
| Type Tags | Compile-time platform identification | `*_platform` structs |
| STL Allocator | Container compatibility | `allocator<T, Memory>` |
| if-constexpr | Zero-cost optional tracking | `Tracking` template param |

---

## Non-Functional Requirements

### Performance

The system SHALL provide zero-overhead abstractions when tracking is disabled and minimal overhead when tracking is enabled.

The system SHALL:
- Add zero runtime overhead when tracking is disabled (`Tracking = false`)
- Use inline methods for allocation hot paths where possible
- Avoid virtual dispatch in type-specific code paths
- Use compile-time dispatch via `if constexpr` instead of runtime checks
- Provide extern template declarations for common instantiations to reduce binary bloat

#### Scenario: Zero-cost abstraction validation
- **WHEN** `allocator<T, host_memory<malloc, false>>` is used
- **THEN** generated assembly is equivalent to direct `malloc()` calls
- **AND** no registration or tracking overhead is present

#### Scenario: Tracking overhead comparable to v1
- **WHEN** tracking is enabled (`Tracking = true`)
- **THEN** per-allocation overhead is comparable to v1 ResourceManager (within 5%)

#### Scenario: Compile-time platform dispatch
- **WHEN** template code uses `if constexpr` with platform tags
- **THEN** branches for non-matching platforms are eliminated at compile time
- **AND** no runtime platform checks occur

---

### Memory Overhead

The system SHALL minimize memory overhead for tracking and registry data structures.

The system SHALL:
- Add O(1) per-allocation overhead for tracking (pointer + size + strategy pointer)
- Use minimal registry metadata (< 1KB per allocator instance)
- Share allocation map with v1 to avoid duplication

#### Scenario: Tracking overhead measurement
- **WHEN** an allocation is tracked
- **THEN** the registry stores at most `sizeof(allocation_record)` bytes per allocation
- **AND** this is comparable to v1 tracking overhead

#### Scenario: Registry memory footprint
- **WHEN** 100 allocator instances are registered
- **THEN** registry metadata uses < 100KB of memory

---

### Compile-Time Overhead

The system SHALL keep template instantiation overhead reasonable and support incremental compilation.

The system SHALL:
- Provide extern template declarations for common instantiations
- Keep template recursion depth reasonable (< 10 levels for typical compositions)
- Support incremental compilation without excessive recompilation

#### Scenario: Compilation time impact
- **WHEN** a translation unit uses `allocator<T, host_memory>`
- **THEN** compile time increase is < 30% vs v1 equivalent

#### Scenario: Template instantiation depth
- **WHEN** strategies are composed (e.g., `thread_safe<fixed_pool<host_memory>>`)
- **THEN** template instantiation depth remains < 10 levels for typical patterns

---

### Portability

The system SHALL maintain C++17 compatibility and support all Umpire target platforms.

The system SHALL:
- Use C++17 standard features only (no C++20 dependencies)
- Support GCC 7+, Clang 8+, MSVC 2019+, and common HPC compilers
- Conditionally compile platform-specific resources based on CMake configuration
- Work on Linux, macOS, and Windows (where applicable)

#### Scenario: C++17 compliance
- **WHEN** the v2 API is compiled with `-std=c++17`
- **THEN** all code compiles without errors or warnings
- **AND** no C++20 features are required

#### Scenario: Platform-specific compilation
- **WHEN** built without CUDA support (`UMPIRE_ENABLE_CUDA=OFF`)
- **THEN** `cuda_device_memory` is not compiled
- **AND** no CUDA headers or libraries are required

---

## Common Usage Patterns

This section provides examples of typical v2 API usage patterns.

### Pattern: Basic host allocation with STL containers

```cpp
#include <umpire/resource/host_memory.hpp>
#include <umpire/allocator.hpp>
#include <vector>

// Get the singleton host memory resource
auto& host_mem = umpire::host_memory<>::get();

// Create a typed allocator for doubles
umpire::allocator<double, umpire::host_memory<>> alloc{&host_mem};

// Use with STL vector
std::vector<double, decltype(alloc)> vec(alloc);
vec.resize(1000);  // Allocates 8000 bytes from host memory
```

### Pattern: GPU allocation with pool

```cpp
#include <umpire/resource/cuda_device_memory.hpp>
#include <umpire/strategy/fixed_pool.hpp>
#include <umpire/allocator.hpp>

// Create a fixed pool backed by CUDA device memory
using pool_t = umpire::fixed_pool<umpire::cuda_device_memory<>>;
auto pool = std::make_shared<pool_t>(
  umpire::cuda_device_memory<>::get(),
  64,    // object size in bytes
  1024   // objects per pool
);

// Create allocator using the pool
umpire::allocator<float, pool_t> gpu_alloc{pool.get()};

// Use with STL containers
std::vector<float, decltype(gpu_alloc)> gpu_vec(gpu_alloc);
gpu_vec.resize(1024);  // Allocated from pre-allocated pool
```

### Pattern: Thread-safe shared allocator

```cpp
#include <umpire/resource/host_memory.hpp>
#include <umpire/strategy/thread_safe.hpp>
#include <umpire/allocator.hpp>
#include <thread>

// Create thread-safe wrapper around host memory
using safe_host_t = umpire::thread_safe<umpire::host_memory<>>;
auto safe_mem = std::make_shared<safe_host_t>(umpire::host_memory<>::get());

// Create allocator
umpire::allocator<int, safe_host_t> safe_alloc{safe_mem.get()};

// Safe to use from multiple threads
auto worker = [&]() {
  std::vector<int, decltype(safe_alloc)> thread_vec(safe_alloc);
  thread_vec.resize(100);
  // Allocations are mutex-protected
};

std::thread t1(worker);
std::thread t2(worker);
t1.join();
t2.join();
```

### Pattern: Strategy composition (thread-safe + pool + size limit)

```cpp
#include <umpire/resource/host_memory.hpp>
#include <umpire/strategy/fixed_pool.hpp>
#include <umpire/strategy/size_limiter.hpp>
#include <umpire/strategy/thread_safe.hpp>
#include <umpire/allocator.hpp>

// Compose strategies: thread-safe wrapper around size-limited fixed pool
using pool_t = umpire::fixed_pool<umpire::host_memory<>>;
using limited_pool_t = umpire::size_limiter<pool_t>;
using safe_limited_pool_t = umpire::thread_safe<limited_pool_t>;

// Create the composed strategy
auto pool = std::make_shared<pool_t>(umpire::host_memory<>::get(), 64, 1024);
auto limited = std::make_shared<limited_pool_t>(pool.get(), 1024 * 1024);  // 1MB limit
auto safe = std::make_shared<safe_limited_pool_t>(limited.get());

// Create allocator
umpire::allocator<char, safe_limited_pool_t> alloc{safe.get()};

// Allocations are:
// 1. Thread-safe (mutex protected)
// 2. Size-limited (throws if > 1MB)
// 3. Pooled (pre-allocated blocks)
```

### Pattern: Cross-platform memory copy

```cpp
#include <umpire/resource/host_memory.hpp>
#include <umpire/resource/cuda_device_memory.hpp>
#include <umpire/allocator.hpp>
#include <umpire/op/copy.hpp>

// Allocate on host
umpire::allocator<float, umpire::host_memory<>> host_alloc{
  &umpire::host_memory<>::get()
};
auto host_ptr = host_alloc.allocate(1000);

// Allocate on GPU
umpire::allocator<float, umpire::cuda_device_memory<>> gpu_alloc{
  &umpire::cuda_device_memory<>::get()
};
auto gpu_ptr = gpu_alloc.allocate(1000);

// Copy host to device using template-based operation
umpire::copy<umpire::host_platform, umpire::cuda_platform>(
  gpu_ptr, host_ptr, 1000 * sizeof(float)
);

// Process on GPU...

// Copy back device to host
umpire::copy<umpire::cuda_platform, umpire::host_platform>(
  host_ptr, gpu_ptr, 1000 * sizeof(float)
);

// Clean up
host_alloc.deallocate(host_ptr, 1000);
gpu_alloc.deallocate(gpu_ptr, 1000);
```

### Pattern: Compile-time platform detection

```cpp
#include <umpire/allocator.hpp>
#include <type_traits>

template <typename Allocator>
void process_data(Allocator& alloc, typename Allocator::size_type n) {
  auto ptr = alloc.allocate(n);

  // Compile-time platform detection
  if constexpr (std::is_same_v<typename Allocator::platform, umpire::cuda_platform>) {
    // CUDA-specific code path
    launch_cuda_kernel(ptr, n);
  } else if constexpr (std::is_same_v<typename Allocator::platform, umpire::host_platform>) {
    // Host-specific code path
    for (size_t i = 0; i < n; ++i) {
      ptr[i] = compute(i);
    }
  }

  alloc.deallocate(ptr, n);
}
```

### Pattern: Zero-overhead allocation (tracking disabled)

```cpp
#include <umpire/resource/host_memory.hpp>
#include <umpire/allocator.hpp>

// Disable tracking for zero overhead
using fast_host = umpire::host_memory<std::allocator<char>, false>;  // Tracking=false
auto& fast_mem = fast_host::get();

umpire::allocator<double, fast_host> fast_alloc{&fast_mem};

// Allocations have zero overhead beyond malloc()
// No registry registration, no statistics tracking
auto ptr = fast_alloc.allocate(10000);
// ... use ptr ...
fast_alloc.deallocate(ptr, 10000);
```

---

## Migration Guide (V1 to V2)

This section provides mappings from v1 API patterns to v2 equivalents.

### V1: Creating an allocator

```cpp
// V1
auto& rm = umpire::ResourceManager::getInstance();
auto alloc = rm.makeAllocator("my_pool",
  rm.makeAllocator("HOST"));
```

```cpp
// V2
auto& host_mem = umpire::host_memory<>::get();
auto alloc = umpire::allocator<char, umpire::host_memory<>>{&host_mem};
```

### V1: Getting an allocator by name

```cpp
// V1
auto alloc = rm.getAllocator("HOST");
```

```cpp
// V2
// Use direct singleton access or registry lookup
auto& host_mem = umpire::host_memory<>::get();
// or
auto* mem = umpire::detail::registry::get().find_by_name("HOST");
```

### V1: Typed allocation

```cpp
// V1
auto alloc = rm.getAllocator("HOST");
double* ptr = static_cast<double*>(alloc.allocate(100 * sizeof(double)));
```

```cpp
// V2
auto& host_mem = umpire::host_memory<>::get();
umpire::allocator<double, umpire::host_memory<>> alloc{&host_mem};
double* ptr = alloc.allocate(100);  // Type-safe, no cast needed
```

### V1: STL containers

```cpp
// V1
auto alloc = rm.getAllocator("HOST");
std::vector<int> vec(umpire::StdAllocator<int>(alloc));
```

```cpp
// V2
auto& host_mem = umpire::host_memory<>::get();
umpire::allocator<int, umpire::host_memory<>> alloc{&host_mem};
std::vector<int, decltype(alloc)> vec(alloc);
```

### V1: Creating a pool

```cpp
// V1
auto alloc = rm.makeAllocator<umpire::strategy::FixedPool>(
  "my_pool", rm.getAllocator("HOST"), 64, 1024);
```

```cpp
// V2
using pool_t = umpire::fixed_pool<umpire::host_memory<>>;
auto pool = std::make_shared<pool_t>(
  umpire::host_memory<>::get(), 64, 1024);
umpire::allocator<char, pool_t> alloc{pool.get()};
```
