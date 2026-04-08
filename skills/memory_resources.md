# Memory Resources

Available resources (platform-dependent):
- HOST: Standard host memory
- DEVICE: GPU device memory (CUDA/HIP/SYCL)
- UM: Unified memory (managed memory)
- PINNED: Page-locked host memory
- DEVICE_CONST: Constant memory on device
- FILE: Memory-mapped files
- SHARED: IPC shared memory
  - SHARED::POSIX (IPC implementation)
  - SHARED::MPI3 (MPI-3 implementation)
  - Use full names when both enabled
- NO_OP: No-op resource (testing/debugging)

Resources are:
- Created by MemoryResourceFactory
- Platform-specific (conditionally compiled)
- Backend abstraction layer

When adding resources:
- Use proper conditional compilation (#ifdef UMPIRE_ENABLE_CUDA)
- Register in MemoryResourceRegistry
- Add corresponding MemoryOperations
- Document platform requirements
