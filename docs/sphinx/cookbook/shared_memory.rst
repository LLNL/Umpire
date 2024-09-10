.. _shared_memory:

=======================
Using IPC Shared Memory 
=======================

Umpire supports the use of IPC Shared Memory on the HOST memory resource. To do this, ``UMPIRE_ENABLE_IPC_SHARED_MEMORY`` 
 should be set to ``On``. Note that you can use IPC Shared Memory with MPI enabled or disabled.

First, to get started with the shared memory allocator, set up the traits. For example:

.. code:: bash
  auto traits{umpire::get_default_resource_traits("SHARED")};

The ``traits`` above is a struct of different properties for your shared allocator. You can
set the maximum size of the allocator with ``traits.size`` and set the scope of the allocator.

For example, you can set the scope to socket:

.. code:: bash
   traits.scope = umpire::MemoryResourceTraits::shared_scope::socket;

However, by default the scope will be set to "node".

Next, create the shared memory allocator:

.. code:: bash
   auto node_allocator{rm.makeResource("SHARED::node_allocator", traits)};

.. note::
   The name of the Shared Memory allocators MUST have "SHARED" in the name. This will help
   Umpire distinguish the allocators as Shared Memory allocators specifically.

Now you can allocate and deallocate shared memory with:

.. code:: bash
   void* ptr{node_allocator.allocate("allocation_name_2", sizeof(uint64_t))};
   ...
   node_allocator.deallocate(ptr);

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Important Notes About Shared Memory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because we are dealing with shared memory there are a few unique characteristics of the Shared Memory allocators
which set it apart from other Umpire allocators.

1. Once you allocate shared memory, that block of memory is fixed. If you need a bigger size, you will have to create a new one.
2. If you want to see how much memory is available for a shared memory allocator, use the ``getActualSize()`` function.
3. File descriptors are used for the shared memory. These files will be under ``/dev/shm``.

There are a few helper functions provided in the ``Umpire.hpp`` header that will be useful when working with 
Shared Memory allocators. For example, you can grab the MPI communicator for a particular Shared Memory allocator with:

.. code:: bash
   MPI_Comm shared_allocator_comm = umpire::get_communicator_for_allocator(node_allocator, MPI_COMM_WORLD);

Note that the ``node_allocator`` is the Shared Memory allocator we created above.
Additionally, we can double check that an allocator has the ``SHARED`` memory resource by asserting:

.. code:: bash
   UMPIRE_ASSERT(node_allocator.getAllocationStrategy()->getTraits().resource ==
                umpire::MemoryResourceTraits::resource_type::shared);

You can see a full example here:
.. literalinclude:: ../../../examples/cookbook/recipe_shared_memory.cpp
