---
name: umpire-core
description: Core Umpire architecture and allocator-semantics guidance. Use when changing ResourceManager, Allocator, AllocationStrategy, MemoryResource, MemoryOperation, introspection, thread-safety, error handling, or performance-sensitive allocation and deallocation paths.
---

# Umpire Core

Preserve Umpire's generic architecture, lightweight allocator model, and hot-path performance constraints. Keep backend-specific logic out of generic layers and use the platform/backend skill when work crosses into CUDA, HIP, SYCL, or platform build details.

## Workflow

1. Read [references/invariants.md](references/invariants.md) first for the non-negotiable architecture, performance, and thread-safety rules.
2. Read [references/components.md](references/components.md) when the task touches allocator behavior, strategies, resources, operations, introspection, or error handling.
3. Pair with `umpire-platform-backends` for backend-specific changes and with `umpire-testing-compatibility` when behavior or API surface changes.

## Operating Rules

- Keep allocators lightweight, copyable handles with no hidden ownership or heavy state.
- Treat allocation and deallocation as hot paths unless the strategy explicitly documents different complexity.
- Avoid hidden synchronization, new global mutable state, and backend-specific leakage into generic headers or generic implementation files.
- Document thread-safety and performance implications whenever behavior changes.
