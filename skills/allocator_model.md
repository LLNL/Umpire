# Allocator Model

Allocator is:
- Lightweight handle (small, copyable)
- Copyable and comparable
- Non-owning (wraps AllocationStrategy)
- O(1) for allocate/deallocate (unless strategy requires otherwise)

Allocator does NOT:
- Own memory pools
- Store raw backend pointers directly
- Perform expensive logic in hot paths
- Have heavy state

Allocators delegate to underlying MemoryResource or Strategy.

Key invariant: Allocators must remain lightweight handles that can be passed by value efficiently.
