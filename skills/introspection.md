# Introspection

Umpire provides introspection capabilities via ResourceManager:

AllocationMap:
- Tracks all live allocations
- Maps pointers to allocation records
- Used for debugging and profiling
- Thread-safe

Query allocation information:
```cpp
auto& rm = umpire::ResourceManager::getInstance();

// Find allocation record
auto record = rm.findAllocationRecord(ptr);
size_t size = record.size;
std::string name = record.name;
std::string allocator_name = record.strategy;

// Get allocator for pointer
auto alloc = rm.getAllocator(ptr);

// Check if pointer is known
bool is_known = rm.hasAllocator(ptr);
```

Allocation records contain:
- Pointer address
- Size in bytes
- Allocator name
- Strategy name (optional)

Allocator introspection:
```cpp
auto alloc = rm.getAllocator("HOST");

// Get statistics
size_t current_size = alloc.getCurrentSize();
size_t high_watermark = alloc.getHighWatermark();
size_t actual_size = alloc.getActualSize();

// Get allocator properties
std::string name = alloc.getName();
int id = alloc.getId();
Platform platform = alloc.getPlatform();
```

Performance considerations:
- AllocationMap lookups have overhead
- May be disabled in release builds (check UMPIRE_ENABLE_LOGGING)
- Do not add unnecessary tracking overhead
- Introspection is for debugging, not hot paths

When modifying tracking:
- Do not alter memory tracking logic without approval
- Maintain thread-safety guarantees
- Document performance impact
- Test with tracking enabled and disabled
