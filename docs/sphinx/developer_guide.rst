.. developer_guide:

===============
Developer Guide
===============

Generating Umpire host-config files
===================================

.. note::
  This is optional if you are on LC machines, since some host-config files have already been generated (at least for Quartz and Lassen) and can be found in the ``host-configs`` repository directory.

Umpire only directly depends on CMake. However, this mechanism will generate a cmake configuration file that reproduces the configuration `Spack <https://github.com/spack/spack>` would have generated in the same context. It contains all the information necessary to build Umpire.

In particular, the host config file will setup:
* flags corresponding with the target required (Release, Debug).
* compilers path, and other toolkits (cuda if required), etc.

This provides an easy way to build Umpire based on `Spack <https://github.com/spack/spack>` and encapsulated in `Uberenv <https://github.com/LLNL/uberenv>`_.

Uberenv role
------------

Uberenv helps by doing the following:

* Pulls a blessed version of Spack locally
* If you are on a known operating system (like TOSS3), we have defined compilers and system packages so you don't have to rebuild the world (CMake typically in Umpire).
* Overrides Umpire Spack packages with the local one if it exists. (see ``scripts/uberenv/packages``).
* Covers both dependencies and project build in one command.

Uberenv will create a directory ``uberenv_libs`` containing a Spack instance with the required Umpire dependencies installed. It then generates a host-config file (``<config_dependent_name>.cmake``) at the root of Umpire repository.

Usage
-----

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

If you already have a spack instance you would like to reuse, you can do so changing the uberenv
command as follow:

.. code-block:: bash

   $ python scripts/uberenv/uberenv.py --upstream=</path/to/my/spack>/opt/spack


