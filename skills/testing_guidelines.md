# Testing Guidelines

Tests should:
- Validate allocator identity and equality
- Validate size correctness
- Validate strategy-specific behavior
- Validate cross-device copies (when GPU enabled)
- Clean up all allocations
- Run quickly (CI constraint)

Avoid:
- Hardcoded device IDs
- Massive memory allocations
- Non-portable timing tests
- Nondeterministic behavior
- Device synchronization (unless testing that behavior)

All changes must build and test in:
- Host-only configuration
- CUDA configuration (if applicable)
- HIP configuration (if applicable)
