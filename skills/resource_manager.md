# ResourceManager

ResourceManager is a singleton that:
- Manages all allocator instances
- Tracks all allocations (via AllocationMap)
- Provides introspection capabilities
- Registers MemoryResources and Operations

Key rules:
- Thread-safe (already implemented)
- Never modify initialization logic without approval
- Do not add global mutable state outside ResourceManager
- Do not change default allocator behavior

Usage pattern:
```cpp
auto& rm = umpire::ResourceManager::getInstance();
auto alloc = rm.getAllocator("HOST");
```
