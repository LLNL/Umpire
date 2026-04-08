# Umpire Architecture

Umpire separates memory management into:

- Allocator (user-facing handle)
- MemoryResource (backend allocation implementation)
- Strategy (policy layer)
- MemoryOperation (copy, move, reallocate)
- ResourceManager (singleton registry)

Design principle:
Frontend API must remain backend-agnostic.
