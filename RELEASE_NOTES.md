# v3.0.0


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
