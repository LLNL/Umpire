# v4.1.1

- Fixed DynamicPoolMap deallocate to make coalesce check O(1) again.

- Initialize m_default_allocator to HOST if not set explicitly.

# v4.1.0

- QuickPool available via the C & Fortran APIs.

- Resources are now created on-demand when accessed for the first time.

- Peer access is no longer automatically enabled for CUDA and HIP.

- Added cmake check to deterime if build subsystem capable of ASAN.

- Fixed ASAN poisoning to limit it to what user originally requested and not
  rounded amount.

- Improved resilliance of primary pool destructors so that giving back
  previously allocated blocks to a device that has already been cleaned up
  will no longer throw an error, but instead will now be logged and ignored.

# v4.0.1

- Fixed Umpire builds with MPI enabled

- Added missing wrapUmpire.hpp to installation directory

# v4.0.0

- Added a FILE memory resource that allocates memory using mmap'd files. This
  can be used to allocate memory from the burst buffers on machines like Sierra
  and Lassen.

- All pools now have an "alignment" parameter that can be provided to the
  constructor.

- MemoryResourceTraits now includes a `resource` member that can be used to
  indentify the underlying resource for any Allocator.

- Bundled tpl cxxopts has been replaced by CLI11 (only used when ENABLE_TOOLS=On)

- Fixed memory leaks in DynamicPoolList, QuickPool.

- Fixed reallocate operation when called on an allocation from a pool.

# v3.0.0

- Added support for multiple GPU devices, detected and registered as "DEVICE_N"
  where N is the device number.

- Added support for capturing function backtraces with allocations.

- Added `AlignedAllocator` to provide aligned allocations for host memory.

- Fixed builds using `-stdlib=c++`

- Switched to camp::Platform: Platform::cpu is now Platform::host

# v2.1.0

- Fixes a bug when calling reallocate with size 0.

- Replay tool now supports replaying reallocate operations.

# v2.0.0

- ENABLE_DEVICE_CONST CMake option to control whether device constant memory
  is enabled. It is now disabled by default.

- DeviceAllocator that provides a pool for allocations inside GPU kernels.

- Added "unset" operations for removing CUDA memory advice.

- Extended C/Fortran API with more allocation strategies.

- NamedAllocator that allows creating a new allocator that passes allocations
  through to underlying strategy

- UMPIRE_VERSION_X are now defined as macros, rather than constexpr variables

- Fixed reallocate to properly handle case where size == 0

- AllocationStrategy constructor parameters re-ordered for consistency

# v1.1.0

- Added symbol `umpire_ver_1_detected` to help detect version mismatches when
  linking multiple libraries that all use Umpire.

- Re-introduced pool algorithm used in pre-1.0.0 releases as `DynamicPoolList`,
  and renamed current strategy to `DynamicPoolMap`. `DynamicPool` is now an
  alias to `DynamicPoolMap`.

- Fix signature of C function `umpire_resourcemanager_make_allocator_pool` to
  take `size_t` not `int`.

- Restored `getActualSize` for all `Allocator` types
# v1.0.1

- Fixed a bug in DynamicPool where memory could be leaked when allocating a new
  block using the "minimum size" for an allocation smaller than the block.

# v1.0.0

- Umpire is MPI-aware (outputs rank information to logs and replays) when
  configured with the option ENABLE_MPI=On, and umpire::initialize(MPI_Comm
  comm) must be called.

- AllocationStrategies may be wrapped with multiple extra layers. To "unwrap" an
  Allocator to a specific strategy, the umpire::util::unwrap_allocator method
  can be used, for example:

  auto dynamic_pool =
    umpire::util::unwrap_allocator<umpire::strategy::DynamicPool>(allocator);

  This will impact users who have been using DynamicPool::coalesce. The cookbook
  recipe has been updated accordingly, and the previous code snippet can be
  used.

- Umpire now directs log and replay output to files, one per process. The
  filenames can be controlled by the environment variable UMPIRE_OUTPUT_BASENAME

- ENABLE_CUDA now set to Off by default.

- Allocations for 0 bytes now always return a valid pointer that cannot be read
  or written. These pointers can be deallocated.
