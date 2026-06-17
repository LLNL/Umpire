.. _no_introspection::

====================================
Introspection Control and Performance
====================================

Umpire provides multiple ways to control introspection overhead, both globally
and per-allocator.

Global Introspection Levels
----------------------------

The simplest way to control introspection is via the global introspection level,
which affects all allocations. Set the ``UMPIRE_INTROSPECTION_LEVEL`` environment
variable to one of:

- ``off``: No tracking at all (maximum performance, zero overhead)
- ``basic``: Lightweight range tracking (fast, but unsafe operations)
- ``on``: Full tracking with safety checks (default)

For example:

.. code-block:: bash

   export UMPIRE_INTROSPECTION_LEVEL=basic
   ./my_application

Or programmatically (must be set before any allocations):

.. code-block:: cpp

   auto& rm = umpire::ResourceManager::getInstance();
   rm.setIntrospectionLevel(umpire::IntrospectionLevel::Basic);

See the :ref:`introspection` tutorial for detailed information on each level.

Per-Allocator Introspection Control
------------------------------------

If you need finer-grained control, you can disable introspection for specific
allocators while keeping it enabled globally. This is useful when you have
one hot path that needs maximum performance but still want introspection elsewhere.

.. note::
    Disabling introspection turns off *all* allocation metadata tracking for
    that Allocator. If you still need exact-pointer ownership tracking but want
    to reduce overhead, consider using the global ``basic`` introspection level via
    ``UMPIRE_INTROSPECTION_LEVEL``.

.. warning::
    Allocations from an allocator with introspection disabled cannot be used
    for operations like ``copy()``, or size and location queries.

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


The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_no_introspection.cpp
