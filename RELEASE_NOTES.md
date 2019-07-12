# v1.0.0

- Umpire is MPI-aware (outputs rank information to logs and replays) when
  configured with the option ENABLE_MPI=On

- AllocationStrategies may be wrapped with mulptile extra layers. To "unwrap" an
  Allocator to a specific strategy, the umpire::util::unwrap_allocator method
  can be used, for example:

  auto dynamic_pool =
    umpire::util::unwrap_allocator<umpire::strategy::DynamicPool>(allocator);

  This will impact users who have been using DynamicPool::coalesce. The cookbook
  recipe has been updated accordingly, and the previous code snippet can be
  used.
