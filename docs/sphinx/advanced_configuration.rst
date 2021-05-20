.. _advanced_configuration:

======================
Advanced Configuration
======================

In addition to the normal options provided by CMake, Umpire uses some additional
configuration arguments to control optional features and behavior. Each
argument is a boolean option, and  can be turned on or off:

.. code-block:: bash

    -DUMPIRE_ENABLE_CUDA=Off

Here is a summary of the configuration options, their default value, and meaning:

    ============================  ======== ===========================================================================
    Variable                      Default  Meaning
    ============================  ======== ===========================================================================
    ``UMPIRE_ENABLE_CUDA``               Off      Enable CUDA support
    ``UMPIRE_ENABLE_HIP``                Off      Enable HIP support
    ``UMPIRE_ENABLE_NUMA``               Off      Enable NUMA support
    ``UMPIRE_ENABLE_FILE_RESOURCE``      Off      Enable FILE support      
    ``UMPIRE_ENABLE_TESTS``              On       Build test executables
    ``UMPIRE_ENABLE_BENCHMARKS``         On       Build benchmark programs
    ``UMPIRE_ENABLE_LOGGING``            On       Enable Logging within Umpire
    ``UMPIRE_ENABLE_SLIC``               Off      Enable SLIC logging
    ``UMPIRE_ENABLE_BACKTRACE``          Off      Enable backtraces for allocations
    ``UMPIRE_ENABLE_BACKTRACE_SYMBOLS``  Off      Enable symbol lookup for backtraces
    ``UMPIRE_ENABLE_TOOLS``              Off      Enable tools like replay
    ``UMPIRE_ENABLE_DOCS``               Off      Build documentation (requires Sphinx and/or Doxygen)
    ``UMPIRE_ENABLE_C``                  Off      Build the C API
    ``UMPIRE_ENABLE_FORTRAN``            Off      Build the Fortran API
    ``UMPIRE_ENABLE_PERFORMANCE_TESTS``  Off      Build and run performance tests
    ``UMPIRE_ENABLE_HOST_SHARED_MEMORY`` Off      Enable Host Shared Memory support
    ``UMPIRE_ENABLE_ASAN``               Off      Enable ASAN support
    ============================  ======== ===========================================================================

These arguments are explained in more detail below:

* ``UMPIRE_ENABLE_CUDA``
  This option enables support for NVIDIA GPUs using the CUDA programming model.
  If Umpire is built without CUDA or HIP support, then only the ``HOST``
  allocator is available for use.

* ``UMPIRE_ENABLE_HIP``
  This option enables support for AMD GPUs using the ROCm stack and HIP
  programming model. If Umpire is built without CUDA or HIP support,
  then only the ``HOST`` allocator is available for use.

* ``UMPIRE_ENABLE_NUMA``
  This option enables support for NUMA. The
  :class:`umpire::strategy::NumaPolicy` is available when built with this
  option, which may be used to locate the allocation to a specific node.

* ``UMPIRE_ENABLE_FILE_RESOURCE``
  This option will allow the build to make all File Memory Allocation files. 
  If Umpire is built without FILE, CUDA or HIP support, then only the ``HOST`` 
  allocator is available for use.

* ``UMPIRE_ENABLE_TESTS``
  This option controls whether or not test executables will be built.

* ``UMPIRE_ENABLE_BENCHMARKS``
  This option will build the benchmark programs used to test performance.

* ``UMPIRE_ENABLE_LOGGING``
  This option enables usage of Logging services for Umpire

* ``UMPIRE_ENABLE_SLIC``
  This option enables usage of logging services provided by SLIC.

* ``UMPIRE_ENABLE_BACKTRACE``
  This option enables collection of backtrace information for each allocation.

* ``UMPIRE_ENABLE_BACKTRACE_SYMBOLS``
  This option enables symbol information to be provided with backtraces.  This
  requires -ldl to be specified for using programs.

* ``UMPIRE_ENABLE_TOOLS``
  Enable development tools for Umpire (replay, etc.)

* ``UMPIRE_ENABLE_DOCS``
  Build user documentation (with Sphinx) and code documentation (with Doxygen)

* ``UMPIRE_ENABLE_C``
  Build the C API, this allows accessing Umpire Allocators and the
  ResourceManager through a C interface.

* ``UMPIRE_ENABLE_FORTRAN``
  Build the Fortran API.

* ``UMPIRE_ENABLE_PERFORMANCE_TESTS``
  Build and run performance tests

* ``UMPIRE_ENABLE_HOST_SHARED_MEMORY``
  This option enables support for interprocess shared memory on the ``HOST``
  platform

* ``UMPIRE_ENABLE_ASAN``
  This option enables address sanitization checks within Umpire by compilers
  that support options like ``-fsanitize=address``
