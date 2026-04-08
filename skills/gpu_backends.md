# GPU Backend Rules

CUDA, HIP, and SYCL support must remain symmetric when possible.

Do not:
- Hardcode CUDA/HIP calls outside conditional compilation
- Assume unified memory is available
- Insert hidden synchronization (no cudaDeviceSynchronize, hipDeviceSynchronize)
- Add backend-specific code in generic layers

Device allocations must:
- Respect active device
- Avoid global device changes
- Avoid blocking/synchronization calls
- Use proper conditional compilation (#ifdef UMPIRE_ENABLE_CUDA, etc.)

Always test that host-only builds still work after GPU changes.
