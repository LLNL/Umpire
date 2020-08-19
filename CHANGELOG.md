# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Develop Branch]

### Added

- Added ASAN memory sanitization to QuickPool

- Added File Memory Allocator

- Added alignment option to QuickPool and DynamicPoolList

- GitHub "action" to check ABI compatibility against `main`

- clang-format file, and CMake infrastructure for styling code

- Added [CLI11](https://github.com/CLIUtils/CLI11) command line parser as a
  built-in third party library.

- Added option to replay to skip replaying of umpire operations

### Changed

- API signature of QuickPool, DynamicPoolList, and DynamicPoolMap are
  now identical to one another.

- Quickpool and DynamicPoolMap now both allocate initial pool block
  lazily like DynamicPoolList.

- GitLab CI pipelines now generate the host-config files on-the-fly.

- GitLab CI pipeline can now trigger pipeline in CHAI with develop version of
  Umpire.

- Bump BLT to v0.3.6

- Applied clang-format to all sources

- Minor updates to fix PGI compiler warnings.

### Removed

- Removed replicated implementations for tracking high watermarks and
  allocated byte counts from the Pools as this is now being provided
  from the AllocationTracker/Inspector

- Final remnants of unused ENABLE_COPY_HEADERS option removed.

- Removed the third party library
  [cxxopts](https://github.com/jarro2783/cxxopts) command line parser as it
  has been replaced by [CLI11](https://github.com/CLIUtils/CLI11).

### Fixed

- Poisoning instrumentation has is now properly balanced between allocate
  and deallocate in DyanmicPoolMap.

- Github action for checking CHANGELOG.

- Type of `auto allocator` in HIP codepath.

- When installing, camp target was not exported.

- Fixed memory leak in DynamicPoolList, QuickPool, and ThreadSafeAllocator 
  tests + replay.

## [3.0.0] - 2020-06-30

### Added

- Add support for multiple CUDA devices. These devices are detected and
  registered as "DEVICE_N", where N is the device number.

- Allocation backtrace may be enabled by building umpire with
  -DENABLE_BACKTRACE

- Umpire exceptions now include backtrace information in the exception string.

- `AlignedAllocator` strategy providing aligned allocations for HOST memory.

- Additional symbol information may be obtained in backtraces with
  -DENABLE_BACKTRACE_SYMBOLS and including `-ldl` for the using program.

- Check for nullptr during `ResourceManager::registerAllocation`.

### Changed

- LC Gitlab CI runs only a subset of targets on PRs, and all of them on main
  and develop branch. Lassen allocation is shorter. Jobs name or more efficient
  to read in UI. All builds goes in `${CI_BUILDS_DIR}/umpire/` to avoid multiple
  directories in `${CI_BUILDS_DIR}`.

- Update BLT to version 0.3.0

- `DeviceAllocator` will issue a trap instruction if it runs out of memory.

- Switched to camp::Platform

### Removed

### Fixed

- In gitlab CI, test jobs fails if no tests were found.

- Clang builds using `-stdlib=libc++` option have been fixed.

## [2.1.0] - 2020-01-30

### Added

- Added replay support for reallocate operations.

### Changed

- LC Gitlab CI now releases quartz resources as soon as possible.

### Removed

### Fixed

- Fixed reallocate corner case of reallocating a zero-sized allocation from a
  previously allocated zero-sized allocation.

## [2.0.0] - 2020-01-13

### Added

- `ENABLE_DEVICE_CONST` CMake option to control whether device constant memory
  is enabled. It is now disabled by default.

- `DeviceAllocator` that provides a pool for allocations inside GPU kernels.

- Added statistic gathering capability to `replay`

- Added "unset" operations for removing CUDA memory advice.

- Extended C/Fortran API with more allocation strategies.

- NamedAllocator that allows creating a new allocator that passes allocations
  through to underlying strategy

- ThreadSafeAllocator added for C/Fortran API. Available using
  UmpireResourceManage `make_allocator_thread_safe` function.

- Replay logs now contain information about operations. These are not yet
  replayed.

- Replay now can display information allocators used in a replay session.

- Replay now can replay `NUMA` and `AllocationPreference` allocations.

- Added `getLargestAvailableBlock` metric to dynamic list and map pools

- Added documentation and recipe for `ThreadSafeAllocator`

### Changed

- LC GitLab CI now using lassen by default instead of butte. Build and test
  split in pairs of jobs for quartz, optimized with `needs` and
  `git_strategy` features.

- Constant device memory is disabled by default.

- `CMAKE_C_STANDARD` is only overridden if it's less than c99.

- Build and install all binaries in the `bin` directory.

- Refactored replay tool implementation in preparation for addition of
  capability to compile replays.

- Replay logs now contain mangled symbols, and these are demangled by the
  `replay` tool.

- Replay tool changed to create a binary index file of the operations from
  the original json file that may be used (and reused) for quicker replays.

- `UMPIRE_VERSION_X` are now defined as macros, rather than constexpr variables

### Removed

- Usage of `__cxa_demangle` in core Umpire library.

### Fixed

- Fixed PGI compiler failures

- Fixed replay test the replay tool as well as validate the output from
  running umpire with REPLAY turned on.

- Fixed compilation errors when `-DENABLE_NUMA=On`.

- Fixed reallocate to properly handle case where size == 0

## [1.1.0] - 2019-09-14

### Added

- Tests for CUDA and HIP replays.

- Test for UMPIRE_LOG_LEVEL environment variable.

- ENABLE_DEVELOPER_DEFAULTS option to set default values during development.

- Add unit tests for the DynamicPool.

- Analysis tool for plotting traces of allocators.

- MixedPool to `allocator_benchmarks.cpp`.

- Add a basic GitLab pipeline testing builds on LC systems.

- CI tests installation.

- The DynamicPool algorithms from 1.0.1 and 0.3.5 are now both available under
  different strategy names: DynamicPoolMap (1.0.1) and DynamicPoolList (0.3.5).

### Changed

- Adjust notifications for CI jobs.

- Use git commit hash as RC version in develop builds.

- Update BLT submodule to fix warnings from CMake 3.14 and warnings from HIP library.

- Generalized Strategy.Device test for all resources.

- Moved `tools/plot_allocations` to `tools/analysis/plot_allocations`.

- Logging output no longer prints to stdout.

- DynamicPool is now an alias to DynamicPoolMap.

- LC GitLab CI now using lassen by default instead of butte.

### Removed

- Extraneous TODOs.

### Fixed

- Bamboo test script and job launch on BLUEOS systems.

- Issue with libNUMA integration and `ResourceManager::move()`.

- Fix signature of C function `umpire_resourcemanager_make_allocator_pool` to
  take size_t not int.

- Restore getActualSize for all Allocator types

## [1.0.1] - 2019-09-04

### Fixed

- Fixed a bug in DynamicPool where memory could be leaked when allocating a new
  block using the "minimum size" for an allocation smaller than the block.

## [1.0.0] - 2019-07-12

### Added

- Added ability to replay allocation maps for testing purposes.

- CI builds for Mac, Linux and Windows via Azure Pipelines

- HCC stage in Docker file.

- GitHub action to automatically delete merged branches.

- Enabled `FixedPool` allocator benchmarks.

- Mixed pool that uses faster fixed pools for smaller allocation sizes,
and a dynamic pool for those that are larger.

- Smoke tests for required third-party libraries.

- `util::FixedMallocPool` for internal use.

- Cookbook for enabling Umpire logging.

- Support for AMD's HIP.

- GCC 4.9 build to Travis CI.

- Added a new IOManager that stores logging and replay output to files.

- Added MPI-awareness to output for both logging and replay.

- `DynamicPool` constructor has a new alignment argument.

- Added HIP build to Travis CI.

- Support for tracked 0-byte allocations across all memory types.

- RELEASE_NOTES file detailing the subset of changes that will impact users the
  most.

### Changed

- Replay program refactored to speed up running of the operations being
  replayed.  New `--time` option added to replay to display operation
  timing information.

- Builds are no longer building tools by default (ENABLE_TOOLS=Off).

- Replay uses JSON format for its I/O.

- OpenMP is off by default.

- CUDA is off by default.

- Switched template parameters to runtime constructor arguments in `FixedPool`.

- Update `README.md` to better describe Umpire capability.

- Update BLT to fix CMake 3.13 warnings and MSVC compatibility.

- `AllocationMap` is significantly faster, and uses `MemoryMap`.

- `ResourceManager` de/registration pass `AllocationRecord` by value.

- `AllocationRecord` struct members are no longer prefixed by `m_`.

- `DynamicPool` directly incorporates Simpool's `DynamicSizePool` and
  uses `FixedMallocPool` internally for a small speedup.

- Added Debug and RelWithDebInfo builds to Travis CI.

- Use unique_ptr internally to ensure cleanup at end of program.

- Use RAII locks with `std::lock_guard`.

- Option ENABLE_WARNINGS_AS_ERRORS now turned off by default.

- `DynamicPool` uses maps underneath for improved performance.

- Add PID to filenames for log and replay output.

- Switch to SPDX licensing.

- Cleaned allocator benchmark code and now use random sizes for DynamicPools.

### Removed

- `ENABLE_ASSERTS` option removed. `UMPIRE_ASSERT` should still be used.

- Merge the remaining classes in Simpool into the core of Umpire.

- Deprecated and unused `replay_allocation_map` tool.

### Fixed

- Fixed bug in replay where it was not correctly replaying AllocationAdvisor
  operations.

- Fixed bug in monotonic pool allocator causing it to always return
  the same allocation.

- Enabled pedantic compiler warnings and fixed errors for GNU, CLANG, INTEL,
  XL, and MSVC compilers.

- YAML file for ReadTheDocs to read in that will cause it to use
  Python 3.7 so that it quits producing build failures when it receives
  a deprecation warning when attempting to run older versions of python.

- Exclude third-party libraries from Doxygen to fix out-of-resources error on
  ReadTheDocs.

- Throw an error if attempting to deallocate with a different Allocator than
  performed the allocation.

- Build on Windows.

- Fixed compilation errors from Intel compiler for newly included third-party
  libraries for json and command line parsing (cxxopts).

- Update calls to allocate_pointer in the FORTRAN bindings to ensure that the
  correct variable type of C_SIZE_T is passed in.  This fixes compile errors in
  IBM XL.

- Fix CodeCov reporting by explicitly downloading older version of upload
  script.

- Fix error where the MemoryMap.inl was not installed.

- Replay and logging files only created when logging/replay are enabled.

- 2019-07-09: Build error with NUMA.

- Issues relating to static initialization of Logger.

## [0.3.5] - 2019-06-11

### Fixed

- Off by one regression introduced in 0.3.4 in
AllocationRecord::AllocationMap::findRecord causing it to incorrectly report
offset of `ptr+size_of_allocation` as found.

## [0.3.4] - 2019-06-06

### Fixed

- Bug AllocationRecord::AllocationMap::findRecord causing it to miss finding
allocations that were zero bytes in length.

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
