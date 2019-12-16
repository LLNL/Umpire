.. _coalesce_pool:

======================
Coalescing Pool Memory
======================

The :class:`umpire::strategy::DynamicPool` provides a
:func:`umpire::strategy::DynamicPool::getLargestAvailableBlock` that may be
used to determine the size of the largest block currently available for
allocation within the pool.
To call this
function, you must get the pointer to the
:class:`umpire::strategy::AllocationStrategy` from the
:class:`umpire::Allocator`:

.. literalinclude:: ../../../examples/cookbook/recipe_get_largest_available_block_in_pool.cpp
                    :lines: 18-24

Once you have the pointer to the appropriate strategy, you can call the
function:

.. literalinclude:: ../../../examples/cookbook/recipe_get_largest_available_block_in_pool.cpp
                    :lines: 32-35

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_get_largest_available_block_in_pool.cpp
