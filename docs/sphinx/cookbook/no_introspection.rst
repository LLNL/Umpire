.. _no_introspection::

=====================
Disable Introspection
=====================

If you know that you won't be using any of Umpire's introspection capabalities
for allocations that come from a particular :class:`umpire::Allocator`, you can
turn off the introspection and avoid the overhead of tracking the associated
metadata.

.. warning::
    Disabling introspection means that allocations from this Allocator cannot
    be used for operations, or size and location queries.

In this recipe, we look at disabling introspection for a pool. To turn off
introspection, you pass a boolean as the second template parameter to the
:func:`umpire::ResourceManager::makeAllocator` method:

.. literalinclude:: ../../../examples/cookbook/recipe_two_no_introspection.cpp
                    :lines: 32-34

Remember that disabling introspection will stop tracking the size of
allocations made from the pool, so the
:func:`umpire::Allocator::getCurrentSize` method will return 0:

.. literalinclude:: ../../../examples/cookbook/recipe_two_no_introspection.cpp
                    :lines: 39


The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_two_no_introspection.cpp
