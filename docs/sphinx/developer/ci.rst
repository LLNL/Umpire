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
