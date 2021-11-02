.. _thread_safety::

=====================
Thread Safe Allocator
=====================

If you want thread-safe access to allocations that come from a particular
:class:`umpire::Allocator`, you can create an instance of a
:class:`umpire::strategy::ThreadSafeAllocator` object that will synchronize access
to it.

In this recipe, we look at creating a :class:`umpire::strategy::ThreadSafeAllocator`
for an :class:`umpire::strategy::QuickPool` object:

.. literalinclude:: ../../../examples/cookbook/recipe_thread_safe.cpp
   :start-after: _sphinx_tag_tut_thread_safe_start
   :end-before: _sphinx_tag_tut_thread_safe_end
   :language: C++

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_thread_safe.cpp
