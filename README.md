# <img src="https://cdn.rawgit.com/LLNL/Umpire/develop/share/umpire/logo/umpire-logo.png" width="128" valign="middle" alt="Umpire"/>  Umpire v6.0.0

[![Travis Build Status](https://travis-ci.com/LLNL/Umpire.svg?branch=develop)](https://travis-ci.com/LLNL/Umpire)
[![Azure Pipelines Build Status](https://dev.azure.com/davidbeckingsale/Umpire/_apis/build/status/LLNL.Umpire?branchName=develop)](https://dev.azure.com/davidbeckingsale/Umpire/_build/latest?definitionId=1&branchName=develop)
[![Documentation Status](https://readthedocs.org/projects/umpire/badge/?version=develop)](https://umpire.readthedocs.io/en/develop/?badge=develop)
[![codecov](https://codecov.io/gh/LLNL/Umpire/branch/develop/graph/badge.svg)](https://codecov.io/gh/LLNL/Umpire) [![Join the chat at https://gitter.im/LLNL/Umpire](https://badges.gitter.im/LLNL/Umpire.svg)](https://gitter.im/LLNL/Umpire?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Umpire is a resource management library that allows the discovery, provision,
and management of memory on machines with multiple memory devices like NUMA and GPUs.

Umpire uses CMake and BLT to handle builds. Since BLT is included as a
submodule, first make sure you run:

    $ git submodule init && git submodule update

Then, make sure that you have a modern compiler loaded, and the configuration is as
simple as:

    $ mkdir build && cd build
    $ cmake ..

CMake will provide output about which compiler is being used. Once CMake has
completed, Umpire can be built with Make:

    $ make

For more advanced configuration you can use standard CMake variables.

# Documentation

Both user and code documentation is available [here](http://umpire.readthedocs.io/).

The Umpire [tutorial](https://umpire.readthedocs.io/en/develop/tutorial.html) provides a step by step introduction to Umpire features.

If you have build problems, we have comprehensive [build system documentation](https://umpire.readthedocs.io/en/develop/advanced_configuration.html) too!

# Poster Reproducibility Information (SC22)

We ran our experiments on LLNL's LC systems (more info can be found [here](https://hpc.llnl.gov/)).

A sequence of steps to replicate our results can be found [here]. This page includes how we built Umpire in order to turn on the Replay tool.

More documentation on the Replay tool itself can be found [here](https://umpire.readthedocs.io/en/develop/sphinx/features/logging and replay.html).

The application we used in our poster experiments can not be shared publicly. However, it was built with NVIDIA GPUs enabled. Thus, the version of Umpire that we built had CUDA enabled. This is documented on our reproducibility appendix page linked above. We also built Umpire with gcc v8.3.1. Otherwise, following the general build steps above will work well. The appendix also gives a brief explanation of how to edit the cmake to include other build commands.

We modified our QuickPool implementation for the experiments we did in the poster. The code modifications are documented in the reproducibility appendix file linked above as well as in the QuickPool source code. QuickPool was modified since that is what our application from the study used. However, similar edits could be made to any of the other pools that Umpire provides. Once you have an application that uses Umpire for memory management, just use this version of Umpire (built with Replay tool enabled), collect a replay file, run that file with the Replay tool, and look at resulting .ult files which can be viewed in pydv - or even Matplotlib. (The reproducibility appendix goes over that process step-by-step!)

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

## Citing Umpire

If you are referencing Umpire in a publication, please use the following citation:

- D. Beckingsale, M. Mcfadden, J. Dahm, R. Pankajakshan and R. Hornung, ["Umpire: Application-Focused Management and Coordination of Complex Hierarchical Memory,"](https://ieeexplore.ieee.org/document/8907404) in IBM Journal of Research and Development. 2019. doi: 10.1147/JRD.2019.2954403

# Release

Umpire is released under an MIT license. For more details, please see the
[LICENSE](./LICENSE) and [RELEASE](./RELEASE) files.

`LLNL-CODE-747640`
`OCEC-18-031`
