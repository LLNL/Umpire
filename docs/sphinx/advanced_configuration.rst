.. _advanced_configuration:

======================
Advanced Configuration
======================

In addition to the normal options provided by CMake, Umpire uses some additional
configuration arguments to control optional features and behavior. Each
argument is a boolean option, and  can be turned on or off:

.. code-block:: bash

    -DENABLE_CUDA=Off

Here is a summary of the configuration options, their default value, and meaning:

    ============================  ======== ===========================================================================
    Variable                      Default  Meaning
    ============================  ======== ===========================================================================
    ``ENABLE_CUDA``               Off      Enable CUDA support
    ``ENABLE_HIP``                Off      Enable HIP support
    ``ENABLE_NUMA``               Off      Enable NUMA support
    ``ENABLE_FILE_RESOURCE``      Off      Enable FILE support      
    ``ENABLE_TESTS``              On       Build test executables
    ``ENABLE_BENCHMARKS``         On       Build benchmark programs
    ``ENABLE_LOGGING``            On       Enable Logging within Umpire
    ``ENABLE_SLIC``               Off      Enable SLIC logging
    ``ENABLE_BACKTRACE``          Off      Enable backtraces for allocations
    ``ENABLE_BACKTRACE_SYMBOLS``  Off      Enable symbol lookup for backtraces
    ``ENABLE_TOOLS``              Off      Enable tools like replay
    ``ENABLE_DOCS``               Off      Build documentation (requires Sphinx and/or Doxygen)
    ``ENABLE_C``                  Off      Build the C API
    ``ENABLE_FORTRAN``            Off      Build the Fortran API
    ``ENABLE_PERFORMANCE_TESTS``  Off      Build and run performance tests
    ``ENABLE_HOST_SHARED_MEMORY`` Off      Enable Host Shared Memory support
    ``ENABLE_ASAN``               Off      Enable ASAN support
    ============================  ======== ===========================================================================

These arguments are explained in more detail below:

* ``ENABLE_CUDA``
  This option enables support for NVIDIA GPUs using the CUDA programming model.
  If Umpire is built without CUDA or HIP support, then only the ``HOST``
  allocator is available for use.

* ``ENABLE_HIP``
  This option enables support for AMD GPUs using the ROCm stack and HIP
  programming model. If Umpire is built without CUDA or HIP support,
  then only the ``HOST`` allocator is available for use.

* ``ENABLE_NUMA``
  This option enables support for NUMA. The
  :class:`umpire::strategy::NumaPolicy` is available when built with this
  option, which may be used to locate the allocation to a specific node.

* ``ENABLE_FILE_RESOURCE``
  This option will allow the build to make all File Memory Allocation files. 
  If Umpire is built without FILE, CUDA or HIP support, then only the ``HOST`` 
  allocator is available for use.

* ``ENABLE_TESTS``
  This option controls whether or not test executables will be built.

* ``ENABLE_BENCHMARKS``
  This option will build the benchmark programs used to test performance.

* ``ENABLE_LOGGING``
  This option enables usage of Logging services for Umpire

* ``ENABLE_SLIC``
  This option enables usage of logging services provided by SLIC.

* ``ENABLE_BACKTRACE``
  This option enables collection of backtrace information for each allocation.

* ``ENABLE_BACKTRACE_SYMBOLS``
  This option enables symbol information to be provided with backtraces.  This
  requires -ldl to be specified for using programs.

* ``ENABLE_TOOLS``
  Enable development tools for Umpire (replay, etc.)

* ``ENABLE_DOCS``
  Build user documentation (with Sphinx) and code documentation (with Doxygen)

* ``ENABLE_C``
  Build the C API, this allows accessing Umpire Allocators and the
  ResourceManager through a C interface.

* ``ENABLE_FORTRAN``
  Build the Fortran API.

* ``ENABLE_PERFORMANCE_TESTS``
  Build and run performance tests

* ``ENABLE_HOST_SHARED_MEMORY``
  This option enables support for interprocess shared memory on the ``HOST``
  platform

* ``ENABLE_ASAN``
  This option enables address sanitization checks within Umpire by compilers
  that support options like ``-fsanitize=address``
