# Backend Rules

## Generic Separation

- Do not put CUDA, HIP, SYCL, or other backend-specific types into generic headers unless that layer is already backend-aware by design.
- Keep backend-specific code in backend-aware resources, allocators, and operations.
- Preserve host-only builds by guarding backend code with the appropriate compile-time checks.

## CUDA, HIP, and SYCL

- Respect the active device when allocating or performing backend operations.
- Do not insert hidden synchronization such as `cudaDeviceSynchronize()` or `hipDeviceSynchronize()`.
- Do not add undocumented stream synchronization.
- Keep CUDA and HIP behavior symmetric where the feature set allows it.
- Treat SYCL as its own backend with explicit conditional compilation and queue-aware semantics.

## Memory Semantics

- Device pointers are not host-accessible.
- Unified memory support is platform-dependent and must not be assumed everywhere.
- Pinned memory is host-accessible but page-locked.
- Constant memory semantics are backend-specific and must not be silently changed.
- Route cross-resource copies and similar actions through the `MemoryOperation` registry rather than ad hoc logic in generic layers.

## OpenMP Target and NUMA

- Treat OpenMP target offload as distinct from plain OpenMP.
- Use `UMPIRE_ENABLE_OPENMP_TARGET` for target offload concerns and `UMPIRE_ENABLE_OPENMP` for standard OpenMP support.
- Keep NUMA support conditionally compiled and platform-aware.

## Practical Rule

- If the task is mostly "how should this compile or behave on a backend or machine configuration?", use this skill first.
