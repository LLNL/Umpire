.. _no_introspection::

=====================
Disable Introspection
=====================

If you know that you won't be using any of Umpire's introspection capabalities
for allocations that come from a particular :class:`umpire::Allocator`, you can
turn off the introspection and avoid the overhead of tracking the associated
metadata.

For host allocations, Umpire also provides a built-in ``HOST_FAST`` allocator
that uses host memory with tracking disabled:

.. code-block:: C++

   auto host_fast = rm.getAllocator("HOST_FAST");

.. warning::
    Disabling introspection means that allocations from this Allocator cannot
    be used for operations, or size and location queries.

In this recipe, we look at disabling introspection for a pool. To turn off
introspection, you pass a boolean as the second template parameter to the
:func:`umpire::ResourceManager::makeAllocator` method:

.. literalinclude:: ../../../examples/cookbook/recipe_no_introspection.cpp
   :start-after: _sphinx_tag_tut_nointro_start
   :end-before: _sphinx_tag_tut_nointro_end
   :language: C++

Remember that disabling introspection will stop tracking the size of
allocations made from the pool, so the
:func:`umpire::Allocator::getCurrentSize` method will return 0:

.. literalinclude:: ../../../examples/cookbook/recipe_no_introspection.cpp
   :start-after: _sphinx_tag_tut_getsize_start
   :end-before: _sphinx_tag_tut_getsize_end
   :language: C++

If you only need allocator statistics such as current size, high watermark, and
allocation count, but do not need pointer lookup or Umpire operations, create a
resource or allocator with ``umpire::Tracking::StatisticsOnly``:

.. code-block:: C++

   auto host_stats = rm.makeResource("HOST_STATS", umpire::Tracking::StatisticsOnly);


The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_no_introspection.cpp
