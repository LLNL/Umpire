# Testing Patterns

Test organization:
- tests/unit/: Unit tests for individual components
- tests/integration/: End-to-end feature tests
- tests/applications/: Application-level tests

Test requirements:
- Build with host-only configuration
- Build with CUDA enabled (if applicable)
- Build with HIP enabled (if applicable)
- Clean up all allocations
- Avoid nondeterminism
- Run quickly (CI constraint)

Good test practices:
- Test allocator identity and equality
- Verify allocation sizes
- Test strategy-specific behavior
- Test cross-device operations (when GPU enabled)
- Use GTEST/GoogleTest framework
- Avoid hardcoded device IDs
- Avoid massive allocations
- Avoid timing-based tests

Test patterns:
```cpp
// Basic allocation test
auto& rm = umpire::ResourceManager::getInstance();
auto alloc = rm.getAllocator("HOST");
void* ptr = alloc.allocate(100);
ASSERT_NE(ptr, nullptr);
alloc.deallocate(ptr);

// Strategy test
auto pool = rm.makeAllocator<DynamicPoolList>("pool", alloc);
// Test pool-specific behavior
```

When adding tests:
- Add unit tests for new classes/functions
- Add integration tests for features
- Test error conditions
- Test edge cases (zero-byte allocations, nullptr, etc.)
