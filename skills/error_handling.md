# Error Handling

Umpire uses exceptions for error reporting.

Exception types:
- umpire::runtime_error: General runtime errors
- umpire::out_of_memory_error: Allocation failures
- Derived from std::runtime_error

Throwing exceptions:
```cpp
#include "umpire/util/error.hpp"

// Use UMPIRE_ERROR macro
UMPIRE_ERROR(runtime_error, "Error message");
UMPIRE_ERROR(runtime_error, fmt::format("Size: {}", size));
```

Exception rules:
- Use exceptions for error conditions, not control flow
- Do NOT use exceptions in device code
- Avoid exceptions in allocation fast paths when possible
- Document which methods may throw

Logging:
- Umpire has optional logging (UMPIRE_ENABLE_LOGGING)
- Use UMPIRE_LOG for debug information
- Logging may be disabled in release builds
- Do not rely on logging for correctness

Error categories:
- Invalid allocator: Allocator not found or invalid ID/name
- Out of memory: Backend allocation failed
- Invalid argument: Null pointer, invalid size, etc.
- Backend error: CUDA/HIP/SYCL error codes
- Configuration error: Resource not available on platform

Backend error handling:
```cpp
#if defined(UMPIRE_ENABLE_CUDA)
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
  UMPIRE_ERROR(runtime_error,
    fmt::format("cudaMalloc failed: {}", cudaGetErrorString(err)));
}
#endif
```

Best practices:
- Check for errors immediately after backend calls
- Provide informative error messages
- Include relevant context (size, allocator name, etc.)
- Clean up resources before throwing
- Document exception guarantees

Error recovery:
- Umpire provides basic exception safety
- Failed allocations do not leak resources
- ResourceManager state remains consistent after exceptions
- Allocators remain valid after allocation failures

When adding error handling:
- Use UMPIRE_ERROR macro, not raw throw
- Check all backend call return codes
- Provide actionable error messages
- Test error paths
- Document which exceptions may be thrown
