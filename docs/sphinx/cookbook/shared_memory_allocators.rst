.. _shared_memory_allocators_recipe:

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

IPC Shared Memory
-----------------

Umpire supports the use of Inter-Process Communication (IPC) Shared Memory on the HOST memory resource. IPC Shared Memory refers to 
the mechanisms that allow processes to communicate with each other and synchronize their actions and involves a method where multiple 
processes can access a common memory space.

To use Umpire's IPC Shared Memory allocators, the ``UMPIRE_ENABLE_IPC_SHARED_MEMORY`` flag 
should be set to ``On``. Note that you can use IPC Shared Memory with MPI enabled or disabled.

First, to get started with the shared memory allocator, set up the traits. For example:

.. code-block:: cpp

    auto traits{umpire::get_default_resource_traits("SHARED")};

The ``traits`` above is a struct of different properties for your shared allocator. You can
set the maximum size of the allocator with ``traits.size`` and set the scope of the allocator.

For example, you can set the scope to socket:

.. code-block:: cpp

   traits.scope = umpire::MemoryResourceTraits::shared_scope::socket;

However, by default the scope will be set to "node".

Next, create the shared memory allocator:

.. code-block:: cpp

   auto node_allocator{rm.makeResource("SHARED::node_allocator", traits)};

.. note::
   The name of the Shared Memory allocators MUST have "SHARED" in the name. This will help
   Umpire distinguish the allocators as Shared Memory allocators. It is also used for discovery 
   by other ranks on node.

Now you can allocate and deallocate shared memory with:

.. code-block:: cpp

   void* ptr{node_allocator.allocate("allocation_name_2", sizeof(uint64_t))};
   ...
   node_allocator.deallocate(ptr);

.. note::
   A name is required in order to allocate memory with IPC Shared Memory allocators. However, if that isn't feasible, you
   can instead use the :class:`umpire::strategy::NamingShim` strategy. This allows you to call allocate with only 1 argument
   for the size in bytes. Check out the :doc:`cookbook recipe <../cookbook/naming_shim>` to learn more.

See the bottom of this page for a full example of how to use Shared Memory Allocators with Umpire.

MPI3 Shared Memory
------------------

In addition to IPC Shared Memory, Umpire also supports MPI3 Shared Memory on the HOST memory resource. As the name suggests, this allocator
uses the MPI3 API for its Shared Memory mechanisms that allow processes to communicate with each other and synchronize their actions.

To use Umpire's MPI3 Shared Memory allocators, the ``UMPIRE_ENABLE_MPI3_SHARED_MEMORY`` flag 
should be set to ``On``. Note that if you are using MPI3 Shared Memory, then MPI must be enabled.

See the bottom of this page for a full example of how to use Shared Memory Allocators with Umpire.

Full IPC Shared Memory Recipe
-----------------------------

The following example shows how to create, use, and destruct the IPC Shared Memory Allocator whether using MPI or not. For an example of using
MPI3 Shared Memory Allocators, users could simply update the included header file. No other code changes are necessary assuming the input
parameter specifies MPI for the following example code.

.. literalinclude:: ../../../examples/cookbook/recipe_shared_memory.cpp
   :language: cpp
