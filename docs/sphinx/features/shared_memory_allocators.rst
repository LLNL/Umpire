.. _shared_memory_allocators:

========================
Shared Memory Allocators
========================

Umpire provides two different kinds of Shared Memory capabilities.
First, Umpire provides Inter-Process Communication (IPC) Shared Memory
which can be used with or without MPI. Secondly, Umpire provides
MPI3 Shared Memory which requires MPI3. Although both kinds of Shared
Memory provide a convenient way to share memory across nodes/sockets,
each type has a few unique characteristics and usage details which
will be outlined in this section of the documentation.

This :doc:`cookbook recipe <../cookbook/shared_memory_allocators>` shows how to use both the IPC and MPI3 Shared Memory allocators.

Important Notes About Shared Memory Allocators
----------------------------------------------

Because we are dealing with shared memory there are a few unique characteristics of the Shared Memory allocators
which set it apart from other Umpire allocators.

1. Once you allocate shared memory, that block of memory is fixed. If you need a bigger size, you will have to create a new one.
2. If you want to see how much memory is available for a shared memory allocator, use the ``getActualSize()`` function.
3. File descriptors are used for the shared memory. These files will be under ``/dev/shm``.
4. Although Umpire does not need to have MPI enabled in order to provide IPC Shared Memory, if users wish to associate shared memory with MPI communicators, Umpire will need to be built with MPI enabled. Of course for the MPI3 Shared Memory, MPI is required.
5. It most likely won't make sense to use memory pools with a shared memory allocator. The way shared memory allocators are implemented makes them already kind of pool-like. Since you have to give them a size when you create them, that is basically the "chunk" of memory you have to work with. Then, the shared memory allocator will manage that chunk for you. Therefore, we *do not* recommend that you use pools on top of shared memory allocators.
6. For some LC machines, running Shared Memory Allocators on the login node may produce runtime errors because the login node may not have access to the correct files. If you get an error on the login node, try a compute node instead.
7. MPI3 Shared Memory Allocators only support a `shared_scope` trait of `node`. For IPC Shared Memory, there is an option for either `node` or `socket`.
8. MPI3 Shared Memory Allocators do not need an explicit name during creation like IPC Shared Memory Allocators do.
9. Users can only use one type of Shared Memory Allocator at a time, not both.

There are a few helper functions provided in the ``Umpire.hpp`` header that will be useful when working with 
Shared Memory allocators. For example, you can grab the MPI communicator for a particular Shared Memory allocator with:

.. code-block:: cpp

   MPI_Comm shared_allocator_comm = umpire::get_communicator_for_allocator(node_allocator, MPI_COMM_WORLD);

Note that the ``node_allocator`` is the IPC Shared Memory allocator we created above. You can also refer to
the full example of the IPC Shared Memory cookbook recipe below.

.. warning::
   If you use the ``umpire::get_communicators_for_allocator(...)`` helper function then you MUST
   also call ``umpire::cleanup_cached_communicators()`` function before you call ``MPI_Finalize()``
   in order to avoid memory leaks.

Additionally, we can double check that an allocator has the ``SHARED`` memory resource by asserting:

.. code-block:: cpp

  UMPIRE_ASSERT(node_allocator.getAllocationStrategy()->getTraits().resource == umpire::MemoryResourceTraits::resource_type::shared);

