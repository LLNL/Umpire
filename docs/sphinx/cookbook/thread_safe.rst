.. _thread_safe_allocator::

======================
Thread Safe Allocation
======================

If you want thread-safe access to allocations that come from a particular
:class:`umpire::Allocator`, you can create an instance of a
`umpire::strategy::ThreadSafeAllocator` object that will synchronize access
to it.

In this recipe, we look at creating a `umpire::strategy::ThreadSafeAllocator`
for an `umpire::strategy::DynamicPool` object:

.. literalinclude:: ../../../examples/cookbook/recipe_thread_safe.cpp
                    :lines: 14-21

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_thread_safe.cpp
