# Core Invariants

## Scope

Use this reference for Umpire-wide rules that apply even before you inspect a specific class or strategy.

## Architecture

- Keep the main separation intact:
  - `Allocator`: lightweight user-facing handle
  - `MemoryResource`: backend allocation implementation
  - `AllocationStrategy`: policy layer that may wrap another strategy
  - `MemoryOperation`: copy, memset, prefetch, advise, reallocate, and similar operations
  - `ResourceManager`: singleton registry and coordination point
- Keep frontend APIs backend-agnostic.
- Keep backend-specific logic in backend-aware layers such as `alloc/`, `resource/`, and backend-specific operation implementations.
- Do not change `ResourceManager` initialization logic or default allocator behavior without explicit approval.

## Performance

- Treat `allocate()` and `deallocate()` as hot paths.
- Avoid new heap allocations, `std::function`, unnecessary virtual dispatch, `dynamic_cast`, iostream usage, and hidden synchronization in hot paths.
- Do not add runtime cost to fast paths unless the change is clearly justified by the strategy design.
- Avoid exceptions in fast allocation paths when there is an established cheaper path.

## Thread Safety

- `ResourceManager` is already thread-safe; do not weaken that guarantee.
- Do not add new race conditions to allocators or strategies.
- Document the thread-safety guarantees of any strategy you add or modify.
- Avoid new locks in hot paths unless the strategy requires them and the cost is understood.
- Do not introduce new static non-const globals outside `ResourceManager`.

## Common Mistakes

- Accidentally adding hidden device synchronization.
- Breaking host-only builds while touching GPU-related code.
- Introducing new ownership or lifetime rules into `Allocator`.
- Changing allocation tracking or allocator equality semantics without noticing downstream impact.
- Treating debugging or introspection paths as if they were free in release builds.

## When to Escalate

- Ask before changing memory semantics, public behavior, allocator identity semantics, or anything that may affect ABI or hot-path complexity.
