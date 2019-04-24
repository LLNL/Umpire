# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CI builds for Mac, Linux and Windows via Azure Pipelines

- Added HCC stage in Docker file

- GitHub action to automatically delete merged branches

- Mixed pool that uses faster fixed pools for smaller allocation sizes,
and a dynamic pool for those that are larger.

### Changed

- Switched template parameters to runtime constructor arguments in `FixedPool`.

- Replay CI tests temporarily disabled.  This line to be removed once we
  enable the tests again after replay works with Azure pipelines and MixedPool

### Removed

### Fixed

- Exclude third-party libraries from Doxygen to fix out-of-resources error on
  ReadTheDocs
  
- Throw an error if attempting to deallocate with a different Allocator than
  performed the allocation

- Building on Windows

## [0.3.3] - 2019-04-11

### Added

- NUMA strategy (umpire::strategy::NumaPolicy) that allows allocating memory
an specific NUMA nodes.

- Implemented << for Allocator, so that it can be printed directly.

- Update getAllocator methods to print list of available Allocators if the
requested Allocator cannot be found.

- Replay now captures coalesce operations from strategy::DynamicPool so that
these can be replayed.

- The replay tool can produce an output file that can be used to verify the
replayed events are correct.

- Cookbook example for creating a pool in pinned memory using FORTRAN.

- GitHub workflow to check for CHANGELOG updates.

- Ability to print allocation records that only match a predicate,
`print_allocator_records()` to get all records from a specific
allocator, and a cookbook recipe to do that.

- Dockerfile for multi-stage builds. Supports building Umpire with GCC, Clang,
and CUDA.

- GitHub action to run Clang static analysis.

- Replay now includes unique replay ID of the logging process to help
distinguish processes in an multi-process run.

- Umpire replay now takes a "--help" option and displays usage information.

- A const iterator for AllocationMap, a free function to pull out a vector of
allocation records for a specific allocator, and a method to calculate the
relative fragmentation.

### Changed

- Umpire now builds as a single library, libumpire.a or libumpire.so, rather
than having one library per source subdirectory.

- Removed shared_ptr usage entirely. Ownership of objects was never "shared"
and the smart pointers added unecessary overhead.

- Moved CHANGELOG to CHANGELOG.md.

### Removed

- The 'coalesce' method was removed from ResourceManager and now must be
accessed directory. examples/cookbook/recipe_coalesce_pool.cpp shows how to do
this.

### Fixed

- Bug in ResourceManager::copy/memset when given a pointer offset into an
allocation.

- Memory leak in judyL2Array.

- While replay already was recording release operations, the tool was not
actually replaying them.  A fix was implemented so that the replay tool will
now also replay any captured release operations.

- `make docs` used to fail, because the build was setup for Read the Docs. A fix
was implemented so Doxygen and Sphinx can be run locally, for instance to test
out new cookbooks.

- REPLAY previously recorded some operations with multiple print statements
causing REPLAY output from MPI runs to become mixed between multiple ranks.
REPLAY has been modified to output each operation onto a single line.
