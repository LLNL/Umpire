# Project Context

## Purpose
Umpire is a C++ resource management library for application-focused management and coordination of hierarchical memory in HPC applications. It provides portable abstractions for discovering, allocating, and managing memory across CPUs, GPUs, NUMA domains, high-bandwidth memory, shared memory, and other devices. The goals are to offer a consistent allocator interface across backends, support advanced allocation strategies and operations, and serve as a performance-portable building block for higher-level libraries and applications.

## Tech Stack
- Modern C++17 (and C where needed) as the primary implementation languages
- CMake + BLT for builds, configuration, and toolchain abstraction
- Optional backends: CUDA, HIP, SYCL, OpenMP, MPI, and NUMA-based shared memory
- GoogleTest / GoogleMock integrated via BLT for unit and integration tests
- Doxygen and Sphinx (ReadTheDocs) for API and user documentation
- Python-based tooling (e.g., Uberenv) for HPC configuration in some environments

## Project Conventions

### Code Style
- Follow the existing Umpire style in `include/` and `src/` (2-space indentation, brace placement and formatting consistent with current files).
- Prefer modern C++17 features when they improve safety and clarity, but keep code portable across many HPC compilers and platforms.
- Public APIs live under `include/umpire`, with implementations under `src/umpire` and subdirectories.
- Use Umpire utilities for logging and error handling (`UMPIRE_LOG`, `UMPIRE_ERROR`, `UMPIRE_ASSERT`, etc.) instead of ad-hoc logging or `std::cout`.
- Add Doxygen comments for new public classes, functions, and important configuration options.

### Architecture Patterns
- Core abstraction is `ResourceManager` plus a family of allocator strategies; new behavior is usually added as a new allocation strategy or resource rather than special-casing callers.
- Platform-specific logic is isolated behind strategy/resource interfaces and selected via `Platform` or traits, minimizing scattered `#ifdef`s.
- Optional backends (CUDA, HIP, SYCL, OpenMP, MPI, shared memory, etc.) are guarded by CMake options (`UMPIRE_ENABLE_*`) and should compile away cleanly when disabled.
- Specifications for capabilities live under `openspec/specs`, and proposed changes under `openspec/changes`; implementation work should follow approved specs and keep them in sync.

### Testing Strategy
- Tests live under `tests/` and are organized by type (unit, integration, applications, tools, etc.).
- All new features and bug fixes should include tests that exercise the new behavior or reproduce the fixed issue.
- Tests are written with GoogleTest/GoogleMock and wired through BLT/CMake; prefer existing tests as templates when adding new ones.
- Local testing typically uses `UMPIRE_ENABLE_TESTS=On` with `ctest` / `make test`, while CI runs a broader matrix across compilers, platforms, and enabled backends.

### Git Workflow
- Active development happens on the `develop` branch; pull requests should target `develop`.
- Use topic branches with prefixes like `feature/<name>` for new features and `bugfix/<name>` for fixes.
- Keep commits focused and descriptive, referencing relevant issues, specs, or change IDs when applicable.
- All pull requests must pass CI and receive maintainer review before being merged.

## Domain Context
- Target users are HPC applications running on heterogeneous systems (multi-core CPUs, GPUs, many-core accelerators, NUMA and shared-memory configurations).
- Umpire is a foundational memory-management library used by other LLNL and external projects, so API stability and backward compatibility are important.
- Common patterns include pooled/arena allocators, dynamic strategies, and operations (copy, move, set, etc.) across memory spaces, often combined with instrumentation and replay for performance analysis.

## Important Constraints
- The library must remain portable across a wide range of HPC systems, compilers, and MPI/GPU stacks; avoid introducing dependencies that are not broadly available on supercomputing platforms.
- Performance and memory overhead are critical; avoid unnecessary allocations, virtual dispatch, or heavy abstractions in hot paths.
- New features depending on non-ubiquitous technologies (e.g., specific GPU backends) should be optional and disabled by default unless explicitly enabled via CMake.
- The project is MIT-licensed and developed under LLNL policies; contributions must preserve license headers and respect existing governance and release practices.

## External Dependencies
- Build and tooling: CMake, BLT, standard build tools (e.g., Make, Ninja), and Uberenv in some environments.
- GPU/parallel backends: CUDA, HIP/ROCm, SYCL, OpenMP, MPI, and NUMA libraries (such as `libnuma`) depending on configuration.
- Testing: GoogleTest and GoogleMock provided via BLT.
- Documentation and services: Doxygen, Sphinx/ReadTheDocs, and CI systems (GitHub Actions, Bamboo, GitLab CI) that run the Umpire test matrix.
