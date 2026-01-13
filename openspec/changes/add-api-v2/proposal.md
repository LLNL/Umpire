# Change: Add Umpire API v2

## Why

The current Umpire API (v1) has served the HPC community well but has several architectural limitations that constrain performance and usability in modern heterogeneous systems:

**Performance Constraints:**
- Runtime polymorphism through virtual dispatch prevents compile-time optimization of allocation paths
- Monolithic `ResourceManager` singleton creates contention points and limits optimization opportunities
- No mechanism for zero-overhead allocation when tracking is not needed

**Usability Issues:**
- Limited compile-time platform information prevents template-based optimizations
- STL allocator compliance requires awkward adapter patterns
- Strategy composition is complex and error-prone with string-based factory methods
- No type safety in allocation APIs (everything is `void*`)

**Architectural Concerns:**
- Global `ResourceManager` state makes unit testing and isolation difficult
- Factory pattern centralizes control and couples allocator creation to ResourceManager
- No clear path for users to extend with custom platforms or strategies

API v2 addresses these fundamental issues through:
- Registry pattern for lightweight identity/tracking without centralized control
- Template-based design enabling zero-cost abstractions and compile-time dispatch
- STL-compliant typed allocators providing type safety and container integration
- Composable strategies using decorator pattern with explicit type composition
- Self-registration mechanism allowing extension without modifying core classes

## What Changes

**NEW Capabilities:**
- `memory` abstract base class with self-registration and unified tracking interface
- `detail::registry` singleton providing allocator identity and allocation tracking (replacing ResourceManager for v2 APIs)
- `allocator<T, Memory>` STL-compliant template with full type safety
- Platform type tags (`host_platform`, `cuda_platform`, `hip_platform`, `sycl_platform`, `omp_target_platform`) for compile-time dispatch
- `memory_resource<Platform>` template hierarchy for concrete memory sources
- Template-based strategy composition: `thread_safe<>`, `fixed_pool<>`, `dynamic_pool_list<>`, `quick_pool<>`, `monotonic_buffer<>`, `size_limiter<>`, `named<>`
- `null_resource` for testing and dry-run scenarios
- Template-based front-end for memory operations (`copy<>`, `memset<>`, `reallocate<>`, `prefetch<>`)
- Configurable tracking via `Tracking` template parameter with `if constexpr` for zero-cost opt-out

**MODIFIED Capabilities:**
- Memory operations gain template-based dispatch while maintaining existing `MemoryOperation` implementations
- Allocation tracking visible to both v1 and v2 APIs through shared allocation map

**BACKWARD COMPATIBLE:**
- `Allocator` type alias maps to `allocator<char>` for code compatibility
- Existing v1 `ResourceManager` API unchanged and fully functional
- Existing operations in `umpire/op/` continue to work with both v1 and v2 allocations
- Global allocation map shared between v1 and v2 for tool compatibility (replay, debugging, introspection)
- Thread safety opt-in via `thread_safe<>` wrapper (same as v1 behavior)

**NOT CHANGED:**
- Existing v1 API remains stable and supported
- `MemoryOperation` subclasses and `MemoryOperationRegistry` unchanged
- Existing tools (replay, debugging, introspection) work with v2 allocations
- Build system and platform detection mechanisms

## Impact

**Affected specs:**
- **NEW**: `api-v2` (this proposal) - Complete new API surface
- **COEXISTS**: v1 API remains in `ResourceManager` and related classes

**Affected code:**
- `include/umpire/` - New public API headers (v1 headers unchanged)
  - `include/umpire/memory.hpp` - Base class and hierarchy
  - `include/umpire/allocator.hpp` - Typed allocator template
  - `include/umpire/registry.hpp` - Registry interface (detail namespace)
  - `include/umpire/platform.hpp` - Platform type tags and traits
  - `include/umpire/strategy/*.hpp` - Strategy templates
- `src/umpire/` - New implementation files (v1 implementations unchanged)
  - Core implementation for new classes
- `include/umpire/op/` - Template front-ends added (existing ops unchanged)
- `tests/` - Comprehensive test suite for v2 API
  - Unit tests for each component
  - Integration tests for composition
  - STL container compatibility tests
  - Thread safety validation
  - Performance benchmarks

**Migration timeline:**
- **Months 0-4**: Implementation and internal testing
- **Months 4-6**: Beta release with select users, gather feedback
- **Months 6-18**: Parallel operation - v1 and v2 fully supported, v1 marked deprecated in documentation
- **Month 18+**: v1 sunset planning - v2 is recommended API, v1 remains functional but not actively developed
- **Future major version**: v1 removal considered (requires major version bump per semantic versioning)

**Performance impact:**
- **Positive**: Zero-cost abstractions when tracking disabled, compile-time optimizations via templates
- **Neutral**: When tracking enabled, overhead comparable to v1
- **Negative**: Potential binary size increase due to template instantiations (mitigated via extern templates)

**Compatibility guarantees:**
- v1 and v2 allocations interoperable through shared tracking
- No breaking changes to existing v1 API
- Tools see allocations from both APIs
- v1 operations work on v2 allocations
- Incremental migration supported (can mix v1/v2 in same codebase)

**Documentation needs:**
- API reference (Doxygen) for all new classes and templates
- User guide for v2 API concepts and usage patterns
- Migration guide with v1-to-v2 mappings and examples
- Tutorial showing common patterns (STL containers, pooling, GPU allocation)
- Performance characterization and best practices
- Design rationale document explaining architectural choices
