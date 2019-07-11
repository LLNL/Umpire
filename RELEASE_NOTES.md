# Unreleased

- Umpire is MPI-aware (outputs rank information to logs and replays) when
  configured with the option ENABLE_MPI=On

- AllocationStrategies may be wrapped with mulptile extra layers - to "unwrap"
  an Allocator to a specific strategy, the
  umpire::util::unwrap_allocator method can be used. This will impact users who
  have been using DynamicPool::coalesce - the cookbook recipe has been updated
  accordingly.
