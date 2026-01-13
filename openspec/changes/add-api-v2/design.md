# API v2 Design Document

## Context

Umpire v1 was designed around a central `ResourceManager` singleton that:
- Manages all allocator instances through factory methods
- Provides global allocation tracking for debugging and replay
- Implements operation dispatch to platform-specific implementations
- Uses runtime polymorphism (virtual functions) for allocator strategies

This architecture has been successful for Umpire's core mission of managing heterogeneous memory, but faces challenges in modern HPC applications:

**Performance Concerns:**
- Virtual dispatch prevents inlining and compile-time optimization
- Global singleton creates contention in multi-threaded workloads
- No way to opt out of tracking overhead for performance-critical paths
- Template-based codes cannot leverage platform information at compile time

**Usability Challenges:**
- STL allocator requirements conflict with void* allocation interface
- String-based factory methods are error-prone and lack type safety
- Strategy composition requires understanding complex configuration strings
- Testing is difficult due to global singleton state

**Extensibility Limitations:**
- Adding new platforms requires modifying ResourceManager
- Custom allocation strategies must integrate with factory mechanism
- No clean extension points for user-defined memory sources

API v2 addresses these issues through a registry-centered, template-based design that maintains backward compatibility while enabling modern C++ patterns.

## Goals / Non-Goals

**Goals:**
- **Zero-cost abstractions**: Template design with `if constexpr` for optional tracking
- **STL compatibility**: Full compliance with C++ Allocator requirements
- **Type safety**: Template-based allocation with compile-time platform information
- **Composability**: Decorator pattern for clear strategy composition
- **Extensibility**: Self-registration mechanism for user-defined memory sources
- **Backward compatibility**: Coexistence with v1 API and tool compatibility
- **Thread safety**: Explicit control via `thread_safe<>` wrapper

**Non-Goals:**
- **Not replacing v1**: Both APIs coexist, v1 remains supported during transition (12-month deprecation timeline)
- **Not changing operations**: Existing `MemoryOperation` implementations unchanged, only adding template front-ends
- **Not breaking tools**: Replay, debugging, and introspection tools must work with v2 allocations
- **Not requiring C++20**: Maintain C++17 compatibility for HPC platform support
- **Not async allocation API**: Stream-ordered and async patterns deferred to future work

## Key Decisions

### Decision 1: Registry Pattern vs Factory Pattern

**Choice**: Lightweight `detail::registry` for identity/tracking only (not allocator creation)

**Rationale**:
- Memory resources self-construct (singletons via `get()`, or direct instantiation by users)
- Registry provides minimal services: unique ID generation and allocation tracking
- Decentralizes control compared to ResourceManager factory pattern
- Users can create memory sources without going through central authority
- Easier to test isolated components without global state coordination

**Alternatives considered**:
- **Full factory pattern like v1** - Rejected: repeats v1 limitations, centralizes control
- **No registry at all** - Rejected: need global allocation tracking for debugging tools and v1 interoperability
- **Registry with factory methods** - Rejected: partial solution, still couples creation to registry

**Trade-offs**:
- **Pro**: Decentralized, testable, extensible
- **Con**: Users must manage memory source lifetimes explicitly (except singletons)

---

### Decision 2: Template-Based Memory Abstraction

**Choice**: Three-level hierarchy - `memory` base class + `memory_resource<Platform>` template + concrete types

**Rationale**:
- **`memory` base class**: Enables polymorphic tracking, introspection, and type erasure for registry storage
- **`memory_resource<Platform>` template**: Provides compile-time platform type propagation
- **Concrete types** (`host_memory`, `cuda_device_memory`, etc.): Explicit, self-documenting memory sources

This approach combines the benefits of runtime polymorphism (tracking) with compile-time dispatch (optimization).

**Alternatives considered**:
- **Pure virtual interface (no templates)** - Rejected: runtime overhead for all operations, no compile-time platform info
- **Pure templates (no base class)** - Rejected: cannot store heterogeneous memory sources in registry, no type erasure
- **CRTP (Curiously Recurring Template Pattern)** - Rejected: more complex for users, doesn't eliminate vtable for tracking methods

**Trade-offs**:
- **Pro**: Compile-time platform info, zero-cost when tracking disabled, polymorphic when needed
- **Con**: More complex type system, potential template bloat (mitigated via extern templates)

---

### Decision 3: STL Allocator Compliance via `allocator<T, Memory>`

**Choice**: Template allocator with full C++ Allocator concept compliance

**Rationale**:
- Seamless integration with `std::vector`, `std::map`, `std::unordered_map`, etc.
- Type safety: allocation returns `T*`, not `void*`
- Standard interface that C++ developers expect and understand
- Enables `std::allocate_shared` for smart pointers
- Platform type propagates through `Allocator::platform` alias

**Alternatives considered**:
- **Minimal allocator (allocate/deallocate only)** - Rejected: limits interoperability with standard library
- **RAII containers without STL compatibility** - Rejected: forces users into Umpire-specific containers
- **Separate STL adapter class** - Rejected: duplicates effort, users need both APIs

**Trade-offs**:
- **Pro**: Standard compliance, broad compatibility, familiar to C++ developers
- **Con**: Requires all STL type aliases, rebind protocol, increases API surface

---

### Decision 4: Platform Type Tags for Compile-Time Dispatch

**Choice**: Empty tag structs (`host_platform`, `cuda_platform`, etc.) with `platform_for<>` trait

**Rationale**:
- Zero runtime overhead (tags are empty types)
- Enables `if constexpr` for platform-specific code paths
- Clear platform intent in type signatures
- Maps to existing `camp::resources::Platform` enum for operation dispatch
- Extensible: users can define custom platform tags

**Example usage**:
```cpp
template <typename Allocator>
void process(Allocator& alloc) {
  if constexpr (std::is_same_v<typename Allocator::platform, cuda_platform>) {
    // Compile-time CUDA path
  } else {
    // Compile-time CPU path
  }
}
```

**Alternatives considered**:
- **Enum-based runtime platform** - Rejected: runtime cost, no compile-time optimization
- **Macro-based platform selection (#ifdef)** - Rejected: poor type safety, testing difficulties
- **Concepts (C++20)** - Rejected: not portable to all HPC platforms yet (still C++17 requirement)

**Trade-offs**:
- **Pro**: Zero cost, type-safe, compile-time optimization
- **Con**: More template instantiations, requires trait mapping to enum

---

### Decision 5: Optional Tracking via Template Parameter

**Choice**: `Tracking` boolean template parameter with `if constexpr` to conditionally compile tracking code

**Rationale**:
- Users can opt out of tracking for zero overhead in performance-critical code
- Same API surface regardless of tracking enabled/disabled
- Compile-time decision avoids runtime branching
- Default tracking enabled matches v1 behavior for safety

**Example**:
```cpp
// With tracking (default)
host_memory<malloc_allocator, true> tracked;  // Records all allocations

// Without tracking (zero overhead)
host_memory<malloc_allocator, false> fast;  // No tracking, pure malloc/free
```

**Alternatives considered**:
- **Always track** - Rejected: performance concerns for high-frequency allocation workloads
- **Separate tracked/untracked types** - Rejected: API duplication, user confusion
- **Runtime flag** - Rejected: runtime overhead even when disabled

**Trade-offs**:
- **Pro**: Zero overhead when disabled, same API regardless
- **Con**: Double template instantiations (tracked + untracked), debugging harder without tracking

---

### Decision 6: Strategy Composition via Decorator Pattern

**Choice**: Strategies wrap memory resources explicitly in template parameters

**Rationale**:
- Composition is explicit and readable: `thread_safe<fixed_pool<host_memory>>`
- Each wrapper adds exactly one concern (single responsibility)
- Type-safe at compile time (no configuration string parsing)
- Easy to reason about: outermost wrapper is first in call chain
- Platform type propagates through wrappers automatically

**Example**:
```cpp
// GPU memory with fixed pool and thread safety
using pool_t = fixed_pool<cuda_device_memory<>>;
using safe_pool_t = thread_safe<pool_t>;
auto alloc = allocator<float, safe_pool_t>{...};
```

**Alternatives considered**:
- **Builder pattern** - Rejected: runtime overhead for configuration, less explicit
- **Configuration objects (like v1)** - Rejected: loses type safety, runtime composition
- **Variadic template pack** - Rejected: order ambiguity, harder to debug

**Trade-offs**:
- **Pro**: Explicit, type-safe, zero overhead, clear semantics
- **Con**: Verbose for deep nesting (mitigated via type aliases), longer compile times

---

### Decision 7: Coexistence with V1 via Shared Allocation Map

**Choice**: Both v1 and v2 use the same underlying allocation map structure

**Rationale**:
- Debugging tools (replay, introspection) see all allocations regardless of API
- Operations can work on allocations from either API
- No data duplication or synchronization between separate maps
- Gradual migration: users can mix v1/v2 in same application
- Testing can validate both APIs simultaneously

**Implementation**:
- `detail::registry` wraps or directly uses the existing allocation map
- `allocation_record` structure compatible with v1 format
- Registry provides same lookup interface for operations

**Alternatives considered**:
- **Separate tracking for v1/v2** - Rejected: fragments tooling, debugging nightmare
- **V2 only, remove v1** - Rejected: too disruptive for existing users (violates 12-month deprecation plan)
- **V2 uses v1 ResourceManager** - Rejected: couples v2 to v1 implementation, defeats purpose

**Trade-offs**:
- **Pro**: Tool compatibility, gradual migration, unified view
- **Con**: Shared state requires coordination, registry tied to v1 allocation format

---

### Decision 8: Template Front-Ends for Memory Operations

**Choice**: Add template-based front-end APIs while keeping existing `MemoryOperation` implementations unchanged

**Rationale**:
- New API: `copy<SrcPlatform, DstPlatform>(dst, src, size)` uses platform tags
- Template dispatches to existing `MemoryOperationRegistry` using `platform_for<>` trait
- No changes to proven operation implementations (CUDA copy, HIP copy, etc.)
- Type-safe platform specification vs runtime enum
- Backward compatible: v1 operations continue to work

**Example**:
```cpp
// V2 template API
copy<host_platform, cuda_platform>(gpu_ptr, cpu_ptr, bytes);

// Internally maps to:
// MemoryOperationRegistry::find("COPY", {Platform::host, Platform::cuda})
```

**Alternatives considered**:
- **Rewrite operations with templates** - Rejected: high risk, no clear benefit
- **No template front-end, use v1 operations** - Rejected: inconsistent with v2 design philosophy
- **Separate operation registry for v2** - Rejected: duplicates implementations

**Trade-offs**:
- **Pro**: Type safety, consistency with v2 design, low risk (thin wrapper)
- **Con**: Two APIs for same operations (temporary during transition)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ USER CODE                                                        │
│                                                                  │
│  std::vector<T, allocator<T, Memory>>                           │
│  allocator<T, Memory>.allocate(n) → T*                          │
│  copy<SrcPlatform, DstPlatform>(dst, src, size)                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ ALLOCATOR LAYER (V2 API)                                         │
│                                                                  │
│  allocator<T, Memory>                                            │
│   ├─ Memory* memory_       // pointer to memory source          │
│   ├─ platform alias        // compile-time platform type        │
│   ├─ allocate(n) → T*      // type-safe allocation              │
│   └─ deallocate(T*, n)     // type-safe deallocation            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ MEMORY ABSTRACTION LAYER                                         │
│                                                                  │
│  memory (abstract base)                                          │
│   ├─ allocate(size) → void*          [pure virtual]             │
│   ├─ deallocate(void*)                [pure virtual]             │
│   ├─ get_platform() → Platform        [pure virtual]             │
│   ├─ track_allocation(ptr, size)     [protected]                │
│   ├─ untrack_allocation(ptr)         [protected]                │
│   ├─ get_name(), get_id()            [non-virtual introspection]│
│   └─ self-registers with registry on construction               │
│                                                                  │
│  memory_resource<Platform> : memory (template)                   │
│   └─ using platform = Platform                                  │
│                                                                  │
│  allocation_strategy : memory (abstract decorator base)          │
│   └─ wraps another memory* for composition                      │
└──────┬──────────────────────────────┬───────────────────────────┘
       │                              │
       ▼                              ▼
┌──────────────────────┐    ┌────────────────────────────────────┐
│ CONCRETE RESOURCES   │    │ STRATEGIES (decorators)            │
│                      │    │                                    │
│ host_memory<>        │    │ thread_safe<Memory>                │
│ cuda_device_memory<> │    │  └─ adds mutex to wrapped memory   │
│ hip_device_memory<>  │    │                                    │
│ sycl_device_memory<> │    │ fixed_pool<Memory>                 │
│ openmp_target_memory<>│    │  └─ pre-allocates blocks           │
│ null_resource         │    │                                    │
│                      │    │ dynamic_pool_list<Memory>          │
│ Each provides:       │    │  └─ growable pool list             │
│ - static get()       │    │                                    │
│ - allocate/deallocate│    │ quick_pool<Memory>                 │
│ - platform tag       │    │  └─ power-of-2 bin allocator       │
└──────────────────────┘    │                                    │
                            │ monotonic_buffer<Memory>           │
                            │  └─ append-only, bulk release      │
                            │                                    │
                            │ size_limiter<Memory>               │
                            │  └─ enforces allocation quota      │
                            │                                    │
                            │ named<Memory>                      │
                            │  └─ tags allocations for debugging │
                            └────────────────────────────────────┘
       │                              │
       └──────────────┬───────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│ REGISTRY (detail::registry)                                      │
│                                                                  │
│  Meyer's Singleton: registry::get()                              │
│                                                                  │
│  Services:                                                       │
│   ├─ get_id() → unique allocator ID      [thread-safe atomic]   │
│   ├─ register_allocator(memory*)         [on memory construct]  │
│   ├─ find_allocator_by_id(id)            [O(1) lookup]          │
│   ├─ find_allocator_by_name(name)        [O(1) lookup]          │
│   └─ allocation_map: ptr → record        [shared with v1]       │
│                                                                  │
│  allocation_record:                                              │
│   ├─ void* ptr                                                  │
│   ├─ size_t size                                                │
│   ├─ memory* strategy                                           │
│   └─ util::backtrace (if UMPIRE_ENABLE_BACKTRACE)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ MEMORY OPERATIONS (existing, unchanged)                          │
│                                                                  │
│  MemoryOperationRegistry::getInstance()                          │
│   └─ find(op_name, {src_platform, dst_platform})                │
│       → MemoryOperation*                                         │
│                                                                  │
│  MemoryOperation implementations:                                │
│   ├─ GenericReallocateOperation                                 │
│   ├─ HostCopyOperation                                          │
│   ├─ CudaCopyOperation                                          │
│   ├─ HipCopyOperation                                           │
│   └─ ... (existing implementations)                             │
│                                                                  │
│  NEW: Template front-ends                                        │
│   ├─ copy<Src, Dst>(dst, src, size)                             │
│   │    → uses platform_for<Src/Dst>                             │
│   │    → finds operation                                        │
│   │    → invokes operation                                      │
│   ├─ memset<Platform>(ptr, val, size)                           │
│   ├─ reallocate<Platform>(ptr, size)                            │
│   └─ prefetch<Platform>(ptr, size)                              │
└─────────────────────────────────────────────────────────────────┘


COEXISTENCE WITH V1:

┌─────────────────────────────────────────────────────────────────┐
│ V1 API (unchanged, still supported)                              │
│                                                                  │
│  ResourceManager::getInstance()                                  │
│   ├─ makeAllocator(name, params) → Allocator                    │
│   ├─ getAllocator(name) → Allocator                             │
│   └─ uses same allocation_map as registry                       │
│                                                                  │
│  Allocator (type alias for allocator<char> in v2)                │
│   └─ allocate(size) → void*                                     │
│                                                                  │
│  Operations continue to work via MemoryOperationRegistry         │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Examples

### Allocation Flow (with tracking)
```
1. User: vec.resize(100)
         on std::vector<double, allocator<double, cuda_device_memory>>

2. STL: calls allocator.allocate(100)

3. allocator<double, cuda_device_memory>:
    a. Multiplies: 100 * sizeof(double) = 800 bytes
    b. Calls: memory_->allocate(800)

4. cuda_device_memory::allocate(800):
    a. Calls: cudaMalloc(&ptr, 800)
    b. if constexpr (Tracking == true):
         track_allocation(ptr, 800)
    c. Returns: void* ptr

5. memory::track_allocation(ptr, 800):
    a. Creates: allocation_record{ptr, 800, this}
    b. Calls: registry::get().register_allocation(record)
    c. Updates: current_size_ += 800, highwatermark_

6. allocator<double, cuda_device_memory>:
    a. Casts: static_cast<double*>(ptr)
    b. Returns: double* to STL

Total overhead: 1 registry insertion, statistics update
If Tracking=false: Steps 5-6 skipped, zero overhead beyond cudaMalloc
```

### Deallocation Flow
```
1. User: vec.clear()

2. STL: calls allocator.deallocate(ptr, 100)

3. allocator<double, cuda_device_memory>:
    a. Knows size: 100 * sizeof(double) = 800 bytes
    b. Casts: ptr to void*
    c. Calls: memory_->deallocate(ptr)

4. cuda_device_memory::deallocate(ptr):
    a. if constexpr (Tracking == true):
         untrack_allocation(ptr)
    b. Calls: cudaFree(ptr)

5. memory::untrack_allocation(ptr):
    a. Looks up: record = registry::get().find_allocation(ptr)
    b. Updates: current_size_ -= record.size
    c. Removes: record from registry
```

### Cross-Platform Copy
```
1. User: copy<host_platform, cuda_platform>(gpu_ptr, cpu_ptr, 1024)

2. Template function copy<host_platform, cuda_platform>:
    a. Resolves: src_platform = platform_for<host_platform>::value
                                = camp::resources::Platform::host
    b. Resolves: dst_platform = platform_for<cuda_platform>::value
                                = camp::resources::Platform::cuda
    c. Looks up: op = MemoryOperationRegistry::getInstance()
                        .find("COPY", {src_platform, dst_platform})
    d. Invokes: op->transform(gpu_ptr, cpu_ptr, 1024)

3. CudaCopyOperation::transform (existing v1 implementation):
    a. Calls: cudaMemcpy(gpu_ptr, cpu_ptr, 1024, cudaMemcpyHostToDevice)
```

## Risks / Trade-offs

### Risk: Template Bloat
**Description**: Increased binary size due to many template instantiations

**Impact**: Longer compile times, larger binaries

**Mitigation**:
- Provide extern template declarations for common instantiations in .cpp files
- Base class methods (tracking, introspection) are non-template to share code
- Document best practices: use type aliases, limit unique instantiations
- Consider compile-time firewall via PIMPL for complex strategies if needed

**Acceptance criteria**: Binary size increase < 20% for typical applications vs v1

---

### Risk: Complex Error Messages
**Description**: Template errors produce verbose, hard-to-read compiler output

**Impact**: Increased developer time debugging type mismatches

**Mitigation**:
- Extensive use of `static_assert` with clear error messages
- Comprehensive concept documentation (even without C++20 Concepts)
- Examples showing common patterns and type aliases
- "Troubleshooting" section in documentation with common errors
- Future: C++20 Concepts when widely available on HPC platforms

**Acceptance criteria**: User testing shows developers can resolve common errors within 5 minutes

---

### Risk: Learning Curve
**Description**: Template composition syntax more complex than v1 string-based factory

**Impact**: Slower adoption, user confusion

**Mitigation**:
- Provide type aliases for common patterns: `using gpu_host_pool = fixed_pool<thread_safe<host_memory>>`
- Comprehensive tutorial with progression from simple to complex
- Migration guide with direct v1-to-v2 mappings
- Interactive examples in documentation
- Beta period (months 4-6) for feedback before full release

**Acceptance criteria**: Beta users can migrate simple allocators in < 30 minutes

---

### Risk: ABI Stability
**Description**: Template-heavy header-only code exposes implementation details, no ABI

**Impact**: Recompilation required for any changes, versioning challenges

**Mitigation**:
- v2 explicitly documented as header-only, no ABI guarantees
- v1 maintains ABI stability as before (unaffected)
- Base class `memory` uses virtual functions for ABI-stable plugin interface if needed
- Major changes result in Umpire major version bump

**Acceptance criteria**: Documentation clearly states ABI policy, users understand trade-off

---

### Risk: Debugging Difficulty Without Tracking
**Description**: `Tracking=false` disables allocation tracking, making leak detection impossible

**Impact**: Memory leaks harder to diagnose in performance-optimized builds

**Mitigation**:
- Default to `Tracking=true` for safety (opt-in to disable)
- Recommend debug builds use tracking, release builds can disable if profiled
- Documentation warns about trade-off
- Provide lightweight tracking mode (future: count-only without backtrace)

**Acceptance criteria**: Users understand trade-off, can enable tracking for debugging

---

### Risk: Strategy Composition Complexity
**Description**: Deep nesting of strategies creates complex types

**Example**: `thread_safe<size_limiter<fixed_pool<cuda_device_memory<>>>>`

**Impact**: Type errors are verbose, unclear which layer failed

**Mitigation**:
- Type aliases for common patterns
- Each strategy documents its requirements and constraints
- Static assertions at each layer with clear messages
- Visual diagram in docs showing layer order and responsibilities

**Acceptance criteria**: 80% of users need ≤2 layers of composition (measured in beta)

---

### Risk: V1/V2 Confusion During Transition
**Description**: Two APIs doing similar things may confuse users

**Impact**: Users mix APIs incorrectly, unclear which to use

**Mitigation**:
- Clear documentation marking v1 as "legacy" and v2 as "recommended"
- Compiler warnings when using v1 (after initial transition period)
- Migration guide shows side-by-side examples
- Blog post / announcement explaining rationale and timeline
- Active communication during deprecation period

**Acceptance criteria**: < 5% of beta feedback reports confusion about which API to use

---

## Migration Plan

### Phase 1: Implementation (Months 0-4)
**Goal**: Complete v2 API implementation and internal validation

**Deliverables**:
- All core components implemented (`memory`, `registry`, `allocator<>`, strategies)
- Comprehensive test suite (unit, integration, thread safety)
- Internal performance benchmarks
- API documentation (Doxygen)

**Exit criteria**:
- All tests pass on Linux (GCC, Clang), macOS (AppleClang)
- CUDA, HIP, SYCL backends functional (where available)
- Zero-cost abstraction validated via compiler explorer / benchmark
- Code review completed

---

### Phase 2: Beta Release (Months 4-6)
**Goal**: Validate API design with select external users

**Activities**:
- Release v2 API as "beta" in Umpire release (e.g., v2024.08.0)
- Clear documentation that v2 is beta, v1 is stable
- User guide, tutorial, migration guide published
- Beta announcement to Umpire user community
- Gather feedback via GitHub issues, user meetings

**Success metrics**:
- ≥3 external projects try v2 API
- ≥80% positive feedback on API usability
- Identify and address pain points

**Exit criteria**:
- Major API issues resolved
- Documentation updated based on feedback
- Decision to proceed to full release

---

### Phase 3: Parallel Operation (Months 6-18)
**Goal**: Both APIs fully supported, encourage v2 adoption

**Activities**:
- Remove "beta" label from v2 (e.g., Umpire v2025.01.0)
- Mark v1 as "deprecated" in documentation (but fully functional)
- Blog posts, tutorials, conference talks promoting v2
- Maintain both v1 and v2 equally (bug fixes, platform support)
- No new v1 features, new features v2-only

**Success metrics**:
- ≥30% of user projects using v2 by month 12
- No major v2 API changes needed (stable)

**Exit criteria**:
- 18 months elapsed, community comfortable with v2
- Majority of active projects migrated or have migration plan

---

### Phase 4: V1 Sunset (Month 18+)
**Goal**: Phase out v1 while maintaining compatibility

**Activities**:
- v1 marked "deprecated, removal planned" in documentation
- v1 remains functional but not actively developed
- v2 is default for examples, tutorials, documentation
- v1 removal planned for next major version (e.g., Umpire v3.0.0)

**Rollback strategy**:
- If critical v2 issue discovered: fix in v2, not rollback to v1
- v2 can be removed without breaking v1 (additive change)
- Emergency rollback: mark v2 deprecated, focus on v1 (unlikely)

---

## Open Questions

### 1. Should memory operations support async/stream-ordered semantics?
**Context**: CUDA/HIP/SYCL support stream-ordered operations, but v1 API is synchronous

**Options**:
- A: Defer to future work, keep synchronous semantics matching v1
- B: Add optional stream parameter to template operations: `copy<Src, Dst>(dst, src, size, stream)`
- C: Separate async variants: `copy_async<Src, Dst>(dst, src, size, stream)`

**Recommendation**: Option A for initial release, Option C for future work
- Rationale: Async semantics are complex, need careful design for cross-platform support
- Defer until v2 API is stable and user needs are clearer

---

### 2. Should singleton memory resources support custom instances?
**Context**: Singletons (e.g., `host_memory::get()`) are convenient but limit flexibility

**Options**:
- A: Only singletons, no custom instances
- B: Both singletons and constructor for custom instances
- C: No singletons, users always instantiate explicitly

**Recommendation**: Option B (current design)
- Rationale: Singletons for convenience in simple cases, custom instances for testing and advanced use cases
- Trade-off: Users must manage lifetime of custom instances

---

### 3. How should we handle platform-specific extensions (e.g., CUDA unified memory)?
**Context**: Some platforms have unique features not expressible in generic API

**Options**:
- A: Platform-specific subclasses with extended APIs (e.g., `cuda_unified_memory`)
- B: Configuration parameters in resource constructor
- C: Ignore platform-specific features in v2, defer to v1 or future work

**Recommendation**: Option A
- Rationale: Type-safe, explicit, allows users to opt-in to platform features
- Example: `cuda_unified_memory` inherits from `memory_resource<cuda_platform>`, adds `prefetch_to_device()` method

---

### 4. Should registry be injectable for unit testing?
**Context**: Meyer's singleton creates global state, complicates isolated unit tests

**Options**:
- A: Keep singleton, accept global state for simplicity
- B: Add dependency injection: `memory` constructor accepts `registry*`
- C: Compile-time registry selection via template parameter

**Recommendation**: Option A for initial release
- Rationale: Simplicity wins, most testing can use integration tests or process isolation
- Future: If testing pain point emerges, add Option B (backward compatible)

---

### 5. What's the policy on exception safety and guarantees?
**Context**: Spec mentions exceptions but doesn't define strong/basic/no-throw guarantees

**Options**:
- A: Best-effort, no formal guarantees (document what each method does)
- B: All operations provide at least basic exception safety
- C: Strong exception safety where possible, documented per method

**Recommendation**: Option B (basic safety minimum)
- Rationale: HPC codes rarely use exceptions for recovery, but should not leak resources on throw
- Allocate methods: no-throw guarantee OR throw with no side effects (allocation not tracked if allocation fails)
- Deallocate methods: no-throw guarantee (undefined behavior to pass invalid pointer)

---

### 6. Should we support allocation callbacks/hooks like v1 AllocationAdvisor?
**Context**: v1 has hooks for tracking, annotation, advice (e.g., `madvise`)

**Status**: Not specified in current spec

**Recommendation**: Defer to future work, add as strategy wrapper if needed
- Rationale: Can be added as decorator (e.g., `instrumented<Memory>`) without changing core design
- Avoids complexity in initial release

---

## Performance Targets

Based on zero-cost abstraction goals:

| Scenario | Target |
|----------|--------|
| Allocation with tracking disabled | Assembly equivalent to direct malloc/cudaMalloc |
| Allocation with tracking enabled | Overhead ≤ v1 ResourceManager (within 5%) |
| STL container iteration | No overhead vs raw pointer (inlines completely) |
| Platform dispatch via `if constexpr` | Zero runtime overhead, branch eliminated at compile time |
| Strategy composition depth 1-2 | Inlines fully with optimization |
| Strategy composition depth 3+ | May have call overhead, document as advanced usage |
| Binary size increase | ≤ 20% for typical application vs v1 |
| Compile time increase | ≤ 30% for typical application vs v1 |

**Validation**: Benchmarks in `tests/benchmarks/` comparing v1 vs v2 for common patterns

---

## Implementation Phases

Breaking down by dependency order:

### Phase 1A: Foundation (no dependencies)
- Platform type tags (`host_platform`, etc.)
- `platform_for<>` trait mapping to `camp::resources::Platform`
- `allocation_record` struct

### Phase 1B: Registry (depends on 1A)
- `detail::registry` singleton
- Thread-safe ID generation
- Allocator registration (list, name map, ID map)
- Allocation map integration with v1

### Phase 1C: Memory Base (depends on 1B)
- `memory` abstract base class
- Self-registration on construction
- Tracking methods (`track_allocation`, `untrack_allocation`)
- Introspection methods

### Phase 2: Concrete Resources (depends on 1C)
- `memory_resource<Platform>` template
- `host_memory<Allocator, Tracking>`
- `cuda_device_memory<...>` (if CUDA enabled)
- `hip_device_memory<...>` (if HIP enabled)
- `sycl_device_memory<...>` (if SYCL enabled)
- `openmp_target_memory<...>` (if OpenMP enabled)
- `null_resource`

### Phase 3: Strategies (depends on 1C, 2)
- `allocation_strategy` base
- `thread_safe<Memory>`
- `fixed_pool<Memory>`
- `size_limiter<Memory>`
- `dynamic_pool_list<Memory>`
- `quick_pool<Memory>`
- `monotonic_buffer<Memory>`
- `named<Memory>`

### Phase 4: Typed Allocator (depends on 1C)
- `allocator<T, Memory>` template
- STL compliance (all type aliases, methods)
- Platform propagation

### Phase 5: Operations Integration (depends on 1A)
- `copy<SrcPlatform, DstPlatform>()` template
- `memset<Platform>()` template
- `reallocate<Platform>()` template
- `prefetch<Platform>()` template

### Phase 6: Testing & Documentation
- Unit tests for all components
- Integration tests (composition, thread safety, cross-platform)
- STL container tests
- Performance benchmarks
- API documentation
- User guide and tutorial
- Migration guide

---

This design balances performance, usability, and backward compatibility while providing a modern C++ API for heterogeneous memory management in HPC applications.
