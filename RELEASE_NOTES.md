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
