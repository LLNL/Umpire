# Memory Operations

MemoryOperations abstract platform-specific operations:
- Copy (memcpy across memory spaces)
- Memset (initialize memory)
- Reallocate (resize allocations)
- Prefetch (hint data movement)
- Advise (memory access hints)

Operations are:
- Registered in MemoryOperationRegistry
- Selected based on source/destination resource types
- Platform-specific implementations

Operation selection:
```cpp
// ResourceManager automatically selects correct operation
rm.copy(dest_ptr, src_ptr);  // Host-to-Device, Device-to-Host, etc.
rm.memset(ptr, value);
```

Available operations by backend:
- Host: HostCopyOperation, HostMemsetOperation, HostReallocateOperation
- CUDA: CudaCopyOperation, CudaMemsetOperation, CudaAdviseOperation, CudaMemPrefetchOperation
- HIP: HipCopyOperation, HipMemsetOperation, HipAdviseOperation
- SYCL: SyclCopyOperation, SyclMemsetOperation, SyclMemPrefetchOperation
- NUMA: NumaMoveOperation
- OpenMP: OpenMPTargetCopyOperation, OpenMPTargetMemsetOperation

When adding operations:
- Register in MemoryOperationRegistry
- Implement for all relevant backends
- Use proper conditional compilation
- Document synchronization behavior
- Avoid hidden device synchronization
- Handle all source/destination combinations
