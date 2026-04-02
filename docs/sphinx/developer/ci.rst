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
    --target api_v2_operations_tests

  ctest --test-dir build-omptarget \
    -R '^api_v2_operations_tests$' \
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
  - the ``device_gpu_docker`` job (manual ``workflow_dispatch`` only) reuses
    the existing Docker targets (``cuda``, ``cuda13``, ``hip``, ``sycl``) on
    ``radiuss-cuda-runners`` to provide a hardware-capable CI entry point for
    future API v2 device-validation beads (for example ``umpire-4og``,
    ``umpire-728``, and ``umpire-e2h``).

These configurations intentionally stop short of claiming device validation is
complete; they provide a discoverable, reproducible CI path that downstream
beads can extend with concrete CUDA/HIP/SYCL/OpenMP-target API v2 tests on
hardware-capable hosts.
