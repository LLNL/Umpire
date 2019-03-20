.. _coalesce_pool:

======================
Coalescing Pool Memory
======================

The :class:`umpire::strategy::DynamicPool` provides a
:func:`umpire::strategy::DynamicPool::coalesce` that can be used to release
unused memory and allocate a single large block that will be able to satisfy
allocations up to the previously observed high-watermark. To call this
function, you must get the ``std::shared_ptr`` to the
:class:`umpire::strategy::AllocationStrategy` from the
:class:`umpire::Allocator`:

.. literalinclude:: ../../../examples/cookbook/recipe_coalesce_pool.cpp
                    :lines: 31-38

Since the underlying stratetgy could be wrapped by an
:class:`umpire::strategy::AllocationTracker`, we attempt to perform two
different casts.

Once you have the pointer to the appropriate strategy, you can call the
function:

.. literalinclude:: ../../../examples/cookbook/recipe_coalesce_pool.cpp
                    :lines: 41

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_coalesce_pool.cpp
