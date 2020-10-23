.. _coalesce_pool:

======================
Coalescing Pool Memory
======================

The :class:`umpire::strategy::DynamicPool` provides a
:func:`umpire::strategy::DynamicPool::coalesce` that can be used to release
unused memory and allocate a single large block that will be able to satisfy
allocations up to the previously observed high-watermark. To call this
function, you must get the pointer to the
:class:`umpire::strategy::AllocationStrategy` from the
:class:`umpire::Allocator`:

.. literalinclude:: ../../../examples/cookbook/recipe_coalesce_pool.cpp
   :start-after: _umpire_tut_unwrap_strategy_start
   :end-before: _umpire_tut_unwrap_strategy_end
   :language: C++

Once you have the pointer to the appropriate strategy, you can call the
function:

.. literalinclude:: ../../../examples/cookbook/recipe_coalesce_pool.cpp
   :start-after: _umpire_tut_call_coalesce_start
   :end-before: _umpire_tut_call_coalesce_end
   :language: C++

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_coalesce_pool.cpp
