# Umpire v0.1.0

Umpire is a resource management libray that allows the discovery, provision,
and management of memory on next-generation architectures.

Umpire uses CMake and BLT to handle builds. Since BLT is included as a
submoduel, first make sure you run:

    $ git submodule init && git submodule update

Then, make sure that you have a modern compiler loaded, and the configuration is as
simple as:

    $ mkdir build && cd build
    $ cmake

CMake will provide output about which compiler is being used. Once CMake has
completed, Umpire can be built with Make:

    $ make

For more advanced configuration you can use standard CMake variables.
