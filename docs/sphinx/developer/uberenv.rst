.. _uberenv:

=======
Uberenv
=======

This page describes how to generate a cmake configuration file that
reproduces the configuration `Spack <https://github.com/spack/spack>` would
have generated in the same context. It contains all the information necessary
to build Umpire with a specific compiler, compile flags, and build options.

In particular, the host config file will setup:
* flags corresponding with the target required (Release, Debug).
* compilers and other toolkits (cuda if required), etc.
* paths to installed dependencies (CMake is currently the only dependency). 

Uberenv helps by doing the following:

* Pulls a blessed version of Spack locally
* If you are on a known operating system (like TOSS3), we have defined
  compilers and system packages so you don't have to rebuild the world (CMake
  typically in Umpire).
* Overrides Umpire Spack packages with the local one if it exists. (see
  ``scripts/uberenv/packages``).
* Covers both dependencies and project build in one command.

Uberenv will create a directory ``uberenv_libs`` containing a Spack instance
with the required Umpire dependencies installed. It then generates a
host-config file (``<config_dependent_name>.cmake``) at the root of Umpire
repository.

Using Uberenv to generate the host-config file
----------------------------------------------

.. code-block:: bash

  $ python scripts/uberenv/uberenv.py

.. note::
  On LC machines, it is good practice to do the build step in parallel on a
  compute node. Here is an example command: ``srun -ppdebug -N1 --exclusive
  python scripts/uberenv/uberenv.py``

Unless otherwise specified Spack will default to a compiler. It is
recommended to specify which compiler to use: add the compiler spec to the
``--spec`` Uberenv command line option.

On blessed systems, compiler specs can be found in the Spack compiler files
in our repository: ``scripts/uberenv/spack_configs/<System
type>/compilers.yaml``.

Some examples uberenv options:

* ``--spec=%clang@4.0.0``
* ``--spec=%clang@4.0.0+cuda``

This will generate a CMake cache file, named with the system host name,
system type, compiler, and the Spack hash for the build options:

.. code-block:: bash

  hc-quartz-toss_3_x86_64_ib-gcc@8.1.0-fjcjwd6ec3uen5rh6msdqujydsj74ubf.cmake

Using host-config files to build Umpire
---------------------------------------

When a host-config file exists for the desired machine and toolchain, it can
easily be used in the CMake build process:

.. code-block:: bash

  $ mkdir build && cd build
  $ cmake -C <path_to>/hc-quartz-toss_3_x86_64_ib-gcc@8.1.0-fjcjwd6ec3uen5rh6msdqujydsj74ubf.cmake ..
  $ cmake --build -j .
  $ ctest --output-on-failure -T test

It is also possible to use this configuration with the CI script outside of CI:

.. code-block:: bash

  $ HOST_CONFIG=<path_to>/<host-config>.cmake scripts/gitlab/build_and_test.sh