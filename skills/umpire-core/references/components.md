# Core Components

## Allocator and ResourceManager

- `Allocator` must remain a small, copyable, comparable handle.
- `Allocator` delegates to an `AllocationStrategy`; it does not own pools or backend resources directly.
- `ResourceManager` manages allocator instances, allocation tracking, and memory operations.
- Keep allocator IDs unique and allocator names unique.

## Strategies

- Strategies are composable policy layers between `Allocator` and `MemoryResource`.
- Common strategies include pools, advisors, prefetchers, limiters, alignment wrappers, and NUMA policies.
- Document performance complexity and thread-safety when you add or change a strategy.
- Preserve composability; avoid changes that make strategies special-case other layers.

## Memory Resources and Operations

- Memory resources represent platform-specific allocation backends such as host, device, unified, pinned, file-backed, or shared memory.
- Memory operations are selected through the operation registry based on source and destination resource types.
- Do not hardcode copy or memset logic in generic layers when the operation registry already owns that decision.

## Introspection and Tracking

- Allocation tracking is for debugging, profiling, and correctness checks; do not silently change its semantics.
- `ResourceManager` introspection paths may have overhead, so avoid turning them into hidden fast-path dependencies.
- When changing tracking-related behavior, keep thread-safety guarantees intact and document any cost changes.

## Error Handling

- Use Umpire's existing error-reporting style and macros rather than ad hoc `throw` statements.
- Include actionable context such as size, allocator name, pointer role, or backend call failure details.
- Check backend return codes immediately and clean up resources before reporting failure.

## Practical Rule

- If a change touches core semantics and backend mechanics at the same time, use this skill for the architecture constraints and the backend skill for the platform details.
