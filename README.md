# <img src="https://cdn.rawgit.com/LLNL/Umpire/develop/share/umpire/logo/umpire-logo.png" width="128" valign="middle" alt="Umpire"/>  Umpire v0.3.0

[![Build Status](https://travis-ci.com/LLNL/Umpire.svg?branch=develop)](https://travis-ci.com/LLNL/Umpire)
[![Documentation Status](https://readthedocs.org/projects/umpire/badge/?version=develop)](https://umpire.readthedocs.io/en/develop/?badge=develop)
[![codecov](https://codecov.io/gh/LLNL/Umpire/branch/develop/graph/badge.svg)](https://codecov.io/gh/LLNL/Umpire) [![Join the chat at https://gitter.im/LLNL/Umpire](https://badges.gitter.im/LLNL/Umpire.svg)](https://gitter.im/LLNL/Umpire?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Umpire is a resource management library that allows the discovery, provision,
and management of memory on next-generation architectures.

Umpire uses CMake and BLT to handle builds. Since BLT is included as a
submodule, first make sure you run:

    $ git submodule init && git submodule update

Then, make sure that you have a modern compiler loaded, and the configuration is as
simple as:

    $ mkdir build && cd build
    $ cmake

CMake will provide output about which compiler is being used. Once CMake has
completed, Umpire can be built with Make:

    $ make

For more advanced configuration you can use standard CMake variables.

# Documentation

Both user and code documentation is available [here](http://umpire.readthedocs.io/).

The Umpire [tutorial](https://umpire.readthedocs.io/en/develop/tutorial.html) provides a step by step introduction to Umpire features.

If you have build problems, we have comprehensive [build sytem documentation](https://umpire.readthedocs.io/en/develop/advanced_configuration.html) too!

# Getting Involved

Umpire is an open-source project, and we welcome contributions from the community.

## Mailing List

The Umpire mailing list is hosted on Google Groups, and is a great place to ask questions:
- [Umpire Users Google Group](https://groups.google.com/forum/#!forum/umpire-users)

## Contributions

We welcome all kinds of contributions: new features, bug fixes, documentation edits; it's all great!

To contribute, make a [pull request](https://github.com/LLNL/Umpire/compare), with `develop` as the destination branch.
We use Travis to run CI tests, and your branch must pass these tests before being merged.

For more information, see the [contributing guide](https://github.com/LLNL/Umpire/blob/develop/CONTRIBUTING.md).

# Authors

Thanks to all of Umpire's
[contributors](https://github.com/LLNL/Umpire/graphs/contributors).

Umpire was created by David Beckingsale (david@llnl.gov).

# Release

Umpire is released under an MIT license. For more details, please see the
[LICENSE](./LICENSE) and [RELEASE](./RELEASE) files.

`LLNL-CODE-747640`
`OCEC-18-031`
