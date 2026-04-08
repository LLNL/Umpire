# Build and CI

## Option Naming

- Umpire exposes `UMPIRE_ENABLE_*` options such as `UMPIRE_ENABLE_CUDA`, `UMPIRE_ENABLE_HIP`, `UMPIRE_ENABLE_SYCL`, `UMPIRE_ENABLE_OPENMP`, and `UMPIRE_ENABLE_NUMA`.
- Some examples also set BLT-facing `ENABLE_*` options. When documenting or editing build logic, make it clear whether the option is owned by Umpire or by BLT/toolchain setup.
- `UMPIRE_ENABLE_OPENMP_TARGET` is separate from `UMPIRE_ENABLE_OPENMP`.
- `BLT_CXX_STD` must remain at least `c++20`.

## Expected Build Coverage

- Host-only builds must continue to configure, build, and test without CUDA, HIP, or SYCL.
- Backend-specific work should preserve the relevant CUDA, HIP, or SYCL builds.
- Do not assume one backend's build settings apply cleanly to another.

## Platform-Specific Examples

- ROCm builds may require `ROCM_ROOT_DIR`, `HIP_ROOT_DIR`, `CMAKE_HIP_ARCHITECTURES`, and architecture-specific values such as `AMDGPU_TARGETS`.
- CUDA builds may require `CMAKE_CUDA_ARCHITECTURES` and `CUDA_TOOLKIT_ROOT_DIR`.
- On HPC systems, verify loaded modules, compiler paths, and host-config or cache-file expectations before changing build documentation or scripts.

## Uberenv and CI

- Use `scripts/uberenv/uberenv.py` to reproduce CI-oriented dependency resolution and local developer builds.
- CI job specs live under `.gitlab/jobs/`; use them as the source for machine-specific specs and variants.
- If a task changes documented build flows or required variants, update the relevant docs and examples.

## Practical Rule

- Prefer documenting one canonical build path per backend plus a short explanation of the option mapping, rather than scattering similar build recipes in multiple places.
