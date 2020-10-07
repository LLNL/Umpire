.. developer_guide:

===============
Developer Guide
===============

Generating Umpire host-config files
===================================

This mechanism will generate a cmake configuration file that reproduces the configuration `Spack <https://github.com/spack/spack>` would have generated in the same context. It contains all the information necessary to build Umpire with the described toolchain.

In particular, the host config file will setup:
* flags corresponding with the target required (Release, Debug).
* compilers path, and other toolkits (cuda if required), etc.
* paths to installed dependencies. However, Umpire only directly depends on CMake.

This provides an easy way to build Umpire based on `Spack <https://github.com/spack/spack>` itself driven by `Uberenv <https://github.com/LLNL/uberenv>`_.

Uberenv role
------------

Uberenv helps by doing the following:

* Pulls a blessed version of Spack locally
* If you are on a known operating system (like TOSS3), we have defined compilers and system packages so you don't have to rebuild the world (CMake typically in Umpire).
* Overrides Umpire Spack packages with the local one if it exists. (see ``scripts/uberenv/packages``).
* Covers both dependencies and project build in one command.

Uberenv will create a directory ``uberenv_libs`` containing a Spack instance with the required Umpire dependencies installed. It then generates a host-config file (``<config_dependent_name>.cmake``) at the root of Umpire repository.

.. note::
  One common error that comes up when using Uberenv is that the ``uberenv_libs`` folder is out of date. To resolve, make sure this folder is deleted before running new scripts for the first time because this folder needs to be regenerated.

Using Uberenv to generate the host-config file
----------------------------------------------

.. code-block:: bash

  $ python scripts/uberenv/uberenv.py

.. note::
  On LC machines, it is good practice to do the build step in parallel on a compute node. Here is an example command: ``srun -ppdebug -N1 --exclusive python scripts/uberenv/uberenv.py``

Unless otherwise specified Spack will default to a compiler. It is recommended to specify which compiler to use: add the compiler spec to the ``--spec`` Uberenv command line option.

On blessed systems, compiler specs can be found in the Spack compiler files in our repository: ``scripts/uberenv/spack_configs/<System type>/compilers.yaml``.

Some examples uberenv options:

* ``--spec=%clang@4.0.0``
* ``--spec=%clang@4.0.0+cuda``
* ``--prefix=<Path to uberenv build directory (defaults to ./uberenv_libs)>``

It is also possible to use the CI script outside of CI:

.. code-block:: bash

  $ SPEC="%clang@9.0.0 +cuda" scripts/gitlab/build_and_test.sh --deps-only

Building dependencies can take a long time. If you already have a Spack instance you would like to reuse (in supplement of the local one managed by Uberenv), you can do so changing the uberenv command as follow:

.. code-block:: bash

  $ python scripts/uberenv/uberenv.py --upstream=<path_to_my_spack>/opt/spack

Using host-config files to build Umpire
---------------------------------------

When a host-config file exists for the desired machine and toolchain, it can easily be used in the CMake build process:

.. code-block:: bash

  $ mkdir build && cd build
  $ cmake -C <path_to>/lassen-blueos_3_ppc64le_ib_p9-clang@9.0.0.cmake ..
  $ cmake --build -j .
  $ ctest --output-on-failure -T test

It is also possible to use the CI script outside of CI:

.. code-block:: bash

  $ HOST_CONFIG=<path_to>/<host-config>.cmake scripts/gitlab/build_and_test.sh

Using Uberenv to configure and run Leak Sanitizer
-------------------------------------------------

During development, it may be beneficial to regularly check for memory leaks. This will help avoid the possibility of having many memory leaks showing up all at once during the CI tests later on. The Leak Sanitizer can easily be configured from the root directory with:

.. code-block:: bash

  $ srun -ppdebug -N1 --exclusive python scripts/uberenv/uberenv.py --spec="%clang@9.0.0 cxxflags=-fsanitize=address"
  $ cd build
  $ cmake -C <path_to>/hc-quartz-toss_3_x86_64_ib-clang@9.0.0.cmake ..
  $ cmake --build -j
  $ ASAN_OPTIONS=detect_leaks=1 make test
  
If there is a leak in one of the tests, it can be useful to gather more information about what happened and more details about where it happened. One way to do this is to run:

.. code-block:: bash

  $ ASAN_OPTIONS=detect_leaks=1 ctest -T test --output-on-failure
 
Additionally, the Leak Sanitizer can be run on one specific test (in this example, the "replay" tests) with:

.. code-block:: bash

  $ ASAN_OPTIONS=detect_leaks=1 ctest -T test -R replay --output-on-failure

This will configure a build with Clang 9.0.0 and the Leak Sanitizer. Depending on the output given when running the test with the Leak Sanitizer, it may be useful to use ``addr2line -e <./path_to/executable> <address_of_leak>`` to see the exact line the output is referring to.
