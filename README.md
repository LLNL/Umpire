# <img src="https://cdn.rawgit.com/LLNL/Umpire/develop/share/umpire/logo/umpire-logo.png" width="128" valign="middle" alt="Umpire"/> Umpire

[![Documentation Status](https://readthedocs.org/projects/umpire/badge/?version=develop)](https://umpire.readthedocs.io/en/develop/?badge=develop)
[![Github Actions Build Status](https://github.com/LLNL/Umpire/actions/workflows/build.yml/badge.svg)](https://github.com/LLNL/Umpire/actions/workflows/build.yml)

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

The Umpire [tutorial](https://umpire.readthedocs.io/en/develop/sphinx/tutorial.html) provides a step by step introduction to Umpire features.

If you have build problems, we have comprehensive [build system documentation](https://umpire.readthedocs.io/en/develop/sphinx/advanced_configuration.html) too!

# Getting Involved

Umpire is an open-source project, and we welcome contributions from the community.

You can also start an issue for a [bug report](https://github.com/LLNL/Umpire/issues/new?assignees=&labels=&projects=&template=bug_report.md) or [feature request](https://github.com/LLNL/Umpire/issues/new?assignees=&labels=&projects=&template=feature_request.md).

## Mailing List and Slack

The Umpire mailing list is hosted on Google Groups, and is a great place to ask questions:
[Umpire Users Google Group](https://groups.google.com/forum/#!forum/umpire-users)

You can also join our RADIUSS slack group and find the "umpire-users" channel to ask questions.
To be sent an invite to the slack group, email us at [umpire-dev@llnl.gov](mailto:umpire-dev@llnl.gov)

## Contributions

We welcome all kinds of contributions: new features, bug fixes, documentation edits; it's all great!

To contribute, make a [pull request](https://github.com/LLNL/Umpire/compare), with `develop` as the destination branch.
We have a series of tests and a CI pipeline, and your branch must pass all of these tests before being merged.

For more information, see the [contributing guide](https://github.com/LLNL/Umpire/blob/develop/CONTRIBUTING.md) and the
[governance policy](https://github.com/LLNL/Umpire/blob/develop/docs/sphinx/governance.rst).

# Authors

Thanks to all of Umpire's
[contributors](https://github.com/LLNL/Umpire/graphs/contributors).

Umpire was created by David Beckingsale (david@llnl.gov).

## Citing Umpire

If you are referencing Umpire in a publication, please use the following citation:

- D. Beckingsale, M. Mcfadden, J. Dahm, R. Pankajakshan and R. Hornung, ["Umpire: Application-Focused Management and Coordination of Complex Hierarchical Memory,"](https://ieeexplore.ieee.org/document/8907404) in IBM Journal of Research and Development. 2019. doi: 10.1147/JRD.2019.2954403

# Release

Umpire is released under an MIT license. For more details, please see the
[LICENSE](./LICENSE), [RELEASE](./RELEASE), and [COPYRIGHT](./COPYRIGHT) files.

`LLNL-CODE-747640`
`OCEC-18-031`
