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
      ``ENABLE_CUDA``              On       Enable CUDA support
      ``ENABLE_TESTING``           On       Build test executables
      ``ENABLE_BENCHMARKS``        On       Build benchmark programs
      ``ENABLE_LOGGING``           On       Enable Logging within Umpire
      ``ENABLE_SLIC``              Off      Enable SLIC logging
      ``ENABLE_ASSERTS``           On       Enable UMPIRE_ASSERT() within Umpire
      ``ENABLE_IPC``               Off      Enable IPC shared memory resource within Umpire
      ``ENABLE_IPC_MPI3``          Off      Enable MPI3 SHM as an IPC memory resource within Umpire
      ===========================  ======== ===============================================================================

These arguments are explained in more detail below:

* ``ENABLE_CUDA``
  This option enables support for GPUs. If CHAI is built without CUDA support,
  then only the ``CPU`` execution space is available for use.

* ``ENABLE_TESTING``
  This option controls whether or not test executables will be built.

* ``ENABLE_BENCHMARKS``
  This option will build the benchmark programs used to test ``ManagedArray``
  performance.

* ``ENABLE_LOGGING``
  This option enables usage of Logging services for Umpire

* ``ENABLE_SLIC``
  This option enables usage of Logging services provided by SLIC.

* ``ENABLE_ASSERTS``
  Enable assert() within Umpire

* ``ENABLE_IPC``
  Enable interface to IPC shared memory resouces

* ``ENABLE_IPC_MPI3``
  Enable MPI3 SHM as an IPC shared memory resource
