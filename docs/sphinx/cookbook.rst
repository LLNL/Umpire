.. _cookbook:

===============
Umpire Cookbook
===============

This section provides a set of recipes that show you how to accomplish specific
tasks using Umpire. The main focus is things that can be done by composing
different parts of Umpire to achieve a particular use case.

Examples include being able to grow and shrink a pool, constructing Allocators
that have introspection disabled for improved performance, and applying CUDA
"memory advise" to all the allocations in a particular pool.

.. toctree::
   :maxdepth: 1

   cookbook/shrinking_pools.rst
   cookbook/no_introspection.rst
   cookbook/pool_advice.rst
   cookbook/advice_device_id.rst
   cookbook/move_to_managed.rst
   cookbook/dynamic_pool_heuristics.rst
   cookbook/move_between_numa.rst
   cookbook/coalesce_pool.rst
   cookbook/pinned_pool.rst
