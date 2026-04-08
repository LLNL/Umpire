# Backend Semantics

Backend separation rules:
- Generic code must remain backend-agnostic
- Backend-specific code lives in resource/ and alloc/
- No CUDA/HIP/SYCL types in generic headers
- Use conditional compilation for all backend code

CUDA backend:
- Respect active device (cudaGetDevice/cudaSetDevice)
- No implicit synchronization (no cudaDeviceSynchronize)
- No stream synchronization unless documented
- Use cuda_runtime_api.h, not driver API
- Conditional: #ifdef UMPIRE_ENABLE_CUDA

HIP backend:
- Must maintain symmetry with CUDA
- Use hip/hip_runtime.h
- Respect active device (hipGetDevice/hipSetDevice)
- No implicit synchronization
- Conditional: #ifdef UMPIRE_ENABLE_HIP

SYCL backend:
- Use SYCL queue abstraction
- Respect device selection
- Conditional: #ifdef UMPIRE_ENABLE_SYCL

Device memory rules:
- Device pointers are not accessible on host
- Unified memory may not be available on all platforms
- Pinned memory is host-accessible but page-locked
- Constant memory is read-only on device

Cross-device operations:
- Use MemoryOperation registry
- Operations are selected based on source/destination types
- No hardcoded copy logic in generic code

When adding backend code:
- Use proper conditional compilation
- Add symmetric support for CUDA and HIP when possible
- Document device requirements
- Test with all enabled backends
- Never break host-only builds
