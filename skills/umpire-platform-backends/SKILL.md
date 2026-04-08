---
name: umpire-platform-backends
description: Backend, platform, and CI guidance for Umpire. Use when changing CUDA, HIP, SYCL, OpenMP target, NUMA, conditional compilation, platform-specific configuration, or Uberenv and CI reproduction workflows.
---

# Umpire Platform Backends

Preserve backend symmetry where possible, keep host-only builds working, and keep platform details out of generic code. Use this skill for backend-specific rules and build configuration details, not for generic allocator semantics.

## Workflow

1. Read [references/backend-rules.md](references/backend-rules.md) when code touches CUDA, HIP, SYCL, OpenMP target, NUMA, or backend-dependent memory/resource behavior.
2. Read [references/platform-and-ci.md](references/platform-and-ci.md) when the task needs CMake options, platform examples, or CI/Uberenv reproduction.
3. Pair with `umpire-core` for generic architecture constraints and with `umpire-testing-compatibility` when new behavior needs coverage.

## Operating Rules

- Keep generic code backend-agnostic.
- Use conditional compilation consistently and never break host-only builds.
- Avoid hidden device or stream synchronization.
- Explain any build-option changes clearly, especially when Umpire and BLT options interact.
