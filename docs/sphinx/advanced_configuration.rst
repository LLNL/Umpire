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

      ===========================  ======== ===============================================================================
      Variable                     Default  Meaning
      ===========================  ======== ===============================================================================
      ``ENABLE_CUDA``              Off      Enable CUDA support
      ``ENABLE_HIP``               Off      Enable HIP support
      ``ENABLE_HCC``               Off      Enable HCC support
      ``ENABLE_NUMA``              Off      Enable NUMA support
      ``ENABLE_SICM``              Off      Enable SICM support
      ``ENABLE_STATISTICS``        Off      Enable collection of memory statistics
      ``ENABLE_TESTING``           On       Build test executables
      ``ENABLE_BENCHMARKS``        On       Build benchmark programs
      ``ENABLE_LOGGING``           On       Enable Logging within Umpire
      ``ENABLE_SLIC``              Off      Enable SLIC logging
      ``ENABLE_TOOLS``             Off      Enable tools like replay
      ``ENABLE_DOCS``              Off      Build documentation (requires Sphinx and/or Doxygen)
      ``ENABLE_C``                 Off      Build the C API
      ``ENABLE_FORTRAN``           Off      Build the Fortran API
      ===========================  ======== ===============================================================================

These arguments are explained in more detail below:

* ``ENABLE_CUDA``
  This option enables support for NVIDIA GPUs using the CUDA programming model.
  If Umpire is built without CUDA, HCC, or HIP support, then only the ``HOST``
  allocator is available for use.

* ``ENABLE_HIP``
  This option enables support for AMD GPUs using the ROCm stack and HIP
  programming model. If Umpire is built without CUDA, HCC, or HIP support,
  then only the ``HOST`` allocator is available for use.

* ``ENABLE_HCC``
  This option enables support for AMD GPUs using the ROCm stack and HCC
  programming model. If Umpire is built without CUDA, HCC, or HIP support,
  then only the ``HOST`` allocator is available for use.

* ``ENABLE_NUMA``
  This option enables support for NUMA. The
  :class:`umpire::strategy::NumaPolicy` is available when built with this
  option, which may be used to locate the allocation to a specific node.

* ``ENABLE_SICM``
  This option replaces the underlying resources and allocators for ``HOST``
  and CUDA memory with SICM. ``SICM_INCLUDE_PATH``, ``SICM_LIBRARY_PATH``,
  and ``JEMALLOC_LIBRARY_PATH`` will need to be provided to CMake.

* ``ENABLE_STATISTICS``
  This option enables collection of memory statistics. If Umpire is built with
  this option, the Conduit library will also be built.

* ``ENABLE_TESTING``
  This option controls whether or not test executables will be built.

* ``ENABLE_BENCHMARKS``
  This option will build the benchmark programs used to test performance.

* ``ENABLE_LOGGING``
  This option enables usage of Logging services for Umpire

* ``ENABLE_SLIC``
  This option enables usage of logging services provided by SLIC.

* ``ENABLE_TOOLS``
  Enable development tools for Umpire (replay, etc.)

* ``ENABLE_DOCS``
  Build user documentation (with Sphinx) and code documentation (with Doxygen)

* ``ENABLE_C``
  Build the C API, this allows accessing Umpire Allocators and the
  ResourceManager through a C interface.

* ``ENABLE_FORTRAN``
  Build the Fortran API.
