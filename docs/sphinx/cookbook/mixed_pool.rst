.. _mixed_pool:

========================================
Mixed Pool Creation and Algorithm Basics
========================================

This recipe shows how to create a default mixed pool, and one that
might be tailored to a specific application's needs. Mixed pools
allocate in an array of :class:`umpire::strategy::FixedPool` for small
allocations, because these have simpler bookkeeping and are very fast,
and a :class:`umpire::strategy::DynamicPool` for larger allocations.

The class :class:`umpire::strategy::MixedPool` uses a generic choice
of :class:`umpire::strategy::FixedPool` of size 256 bytes to 4MB in
increments of powers of 2, while
:class:`umpire::strategy::MixedPoolImpl` has template arguments that
select the first, power of 2 increment, and last fixed pool size.

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_mixed_pool.cpp
