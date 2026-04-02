.. _ci:

======================
Continuous Integration
======================

Gitlab CI
---------

Umpire uses continuous integration to ensure that changes added to the
repository are well integrated and tested for compatability with the rest
of the existing code base. Our CI tests incude a variety of vetted
configurations that run on different LC machines.

Umpire shares its Gitlab CI workflow with other projects. The documentation is
therefore `shared <https://radiuss-shared-ci.readthedocs.io/en/latest/>`_.

HIP validation hosts
~~~~~~~~~~~~~~~~~~~~

The RADIUSS configuration in this repository also includes ROCm/HIP-capable
machine pipelines under ``scripts/radiuss-spack-configs/gitlab/radiuss-jobs/``.
For example, ``scripts/radiuss-spack-configs/gitlab/radiuss-jobs/corona.yml``
defines a ``rocmcc_5_7_1_hip`` job that builds a +rocm Umpire spec on the
``corona`` LC system. The associated Spack package for Umpire in
``scripts/radiuss-spack-configs/spack_repo/llnl_radiuss/packages/umpire/package.py``
enables HIP via the ``+rocm`` variant and writes CMake cache entries such as
``ENABLE_HIP`` and the appropriate ROCm paths.

In practice, a HIP-capable LC machine wired into this shared CI (for example,
corona via the ``rocmcc_5_7_1_hip`` job) is the natural place to run API v2 HIP
validation. A representative manual validation flow on such a host looks like:

.. code-block:: bash

  # On an LC system with ROCm/HIP (for example, corona with rocmcc HIP modules)
  cmake -S . -B build-hip -G Ninja \
    -DCMAKE_CXX_COMPILER=/opt/rocm-6.4.3/bin/amdclang++ \
    -DROCM_PATH=/opt/rocm-6.4.3 \
    -DENABLE_HIP=On \
    -DUMPIRE_ENABLE_DEVELOPER_DEFAULTS=On \
    -DUMPIRE_ENABLE_TESTS=On

  cmake --build build-hip --parallel \
    --target api_v2_hip_device_memory_tests api_v2_operations_tests

  ctest --test-dir build-hip \
    -R '^(api_v2_hip_device_memory_tests|api_v2_operations_tests)$' \
    --output-on-failure

As with the OpenMP target path, this configuration does not rely on the local
macOS development machine. It records a concrete host path and
configure/test invocation that downstream beads (for example ``umpire-4og``)
can use when exercising API v2 HIP device coverage on a HIP-capable system.

OpenMP target validation hosts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RADIUSS shared CI configuration in this repository includes machine
pipelines under ``scripts/radiuss-spack-configs/gitlab/radiuss-jobs/`` for LC
systems such as ``corona``, ``dane``, ``lassen`` and ``tioga``. The
corresponding Spack package for Umpire, defined in
``scripts/radiuss-spack-configs/spack_repo/llnl_radiuss/packages/umpire/package.py``,
supports an ``+omptarget`` variant and writes a CMake cache entry
``UMPIRE_ENABLE_OPENMP_TARGET`` for OpenMP target builds.

In practice, this means that an LC machine wired into the shared CI (for
example, tioga via ``radiuss-jobs/tioga.yml``) is the natural place to run
API v2 OpenMP target validation. A typical manual validation flow on such a
host looks like:

.. code-block:: bash

  # On an OpenMP target-capable LC system (for example, tioga with CCE modules)
  cmake -S . -B build-omptarget -G Ninja \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_CXX_COMPILER=CC \
    -DENABLE_OPENMP=On \
    -DUMPIRE_ENABLE_OPENMP_TARGET=On \
    -DUMPIRE_ENABLE_TESTS=On

  cmake --build build-omptarget --parallel \
    --target api_v2_openmp_target_memory_tests api_v2_operations_tests

  ctest --test-dir build-omptarget \
    -R '^(api_v2_openmp_target_memory_tests|api_v2_operations_tests)$' \
    --output-on-failure

This configuration does not rely on the local macOS development machine for
OpenMP target support. Instead, it records a concrete host path and
configure/test invocation that downstream beads (for example ``umpire-e2h``
and ``umpire-4og``) can use when exercising ``openmp_target_memory`` and
related API v2 operations on capable hardware.

GitHub CI and device-capable runners
------------------------------------

For API v2 work, this repository also defines GitHub Actions workflows under
``.github/workflows``:

- ``build.yml``:
  - ``build_docker`` job builds the shared Dockerfile targets
    (``gcc``, ``clang``, ``tsan``, ``hip``, ``sycl``, ``intel``) on
    ``radiuss-cpu-runners``.
  - ``build_gpu`` job builds the CUDA Docker targets (``cuda``, ``cuda13``) on
    ``radiuss-cuda-runners``, which are GPU-equipped self-hosted runners.
- ``api_v2.yml``:
  - host-only jobs validate API v2 targets on GitHub-hosted Ubuntu and macOS.
  - the ``device_cuda_validate`` job (manual ``workflow_dispatch`` only) builds
    backend-specific CUDA images (``api_v2_cuda_validate`` and
    ``api_v2_cuda13_validate``) on ``radiuss-cuda-runners`` and runs
    ``api_v2_cuda_device_memory_tests`` plus ``api_v2_operations_tests`` in
    those containers.
  - the ``device_sycl_validate`` job (also manual ``workflow_dispatch``) reuses
    ``radiuss-cpu-runners`` to build the ``api_v2_sycl_validate`` image and run
    ``api_v2_sycl_device_memory_tests`` and ``api_v2_operations_tests`` inside
    that container.

In practice, an operator with access to the GitHub-hosted repository can:

1. Navigate to the repository's "Actions" tab.
2. Select the "API v2 Tests" workflow defined in ``api_v2.yml``.
3. Use "Run workflow" (the manual ``workflow_dispatch`` entry point) on the
   desired branch (for example, ``feature/api-refactor``).
4. After the workflow starts, expand the ``device_cuda_validate`` and
   ``device_sycl_validate`` jobs to observe the API v2 CUDA and SYCL device
   results.

These configurations intentionally stop short of claiming device validation is
complete; they provide a discoverable, reproducible CI path that downstream
beads can extend with concrete CUDA/HIP/SYCL/OpenMP-target API v2 tests on
hardware-capable hosts.
