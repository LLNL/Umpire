.. _thread_safety::

=====================
Thread Safe Allocator
=====================

If you want to utilize the inter-process shared memory, you can create an instance of a
:class:`umpire::strategy::HostSharedMemory` object.

In this recipe, we look at creating a :class:`umpire::strategy::HostSharedMemory` object:

.. literalinclude:: ../../../examples/cookbook/recipe_shared_memory.cpp
   :language: C++
