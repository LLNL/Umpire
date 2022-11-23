.. _coalesce_pool:

======================
Coalescing Pool Memory
======================

Umpire's memory pools provide a more performant way to allocate large
amounts of memory with fewer calls to the underlying API. 
Memory pools allow developers to allocate all needed memory 
at once instead of making multiple, smaller memory allocations which can 
become quite expensive. However, the performance of memory pools 
varies widely depending upon how the memory blocks within the pool are managed.
Each allocation within a memory pool must fit within a `memory block`.
As memory is allocated and deallocated, those blocks must be properly 
adjusted in order to handle new, incoming memory allocations.

Umpire's solution for managing blocks of memory within a pool is to use a
`coalescing function`. The coalescing function will deallocate a certain amount
of unused memory and reallocate it, creating a new memory block which can handle new allocations.

The figure below attempts to describe how a particular pool ("Current Pool") 
will handle a new incoming allocation into the pool with and without coalescing.

.. image:: ./memory-pool-with-without-coalesce.png

As depicted, if the memory pool can coalesce, it will deallocate those "free" (i.e. unused) blocks
of memory and reallocate one, larger memory block that can handle the new allocation.
Otherwise, if the pool can't coalesce, it will need to grow to accomodate the new allocation.
In this particular example, the free blocks are all too small to handle the new
allocation, so they can't be reused to accomodate that new chunk of memory until a
coalesce has happened. If the pool can't coalesce as much as it needs to, it can
grow too big and prematurely run out of memory.

In Umpire, the QuickPool memory pool (:class:`umpire::strategy::QuickPool`) provides a coalescing
function (:func:`umpire::strategy::QuickPool::coalesce`) that can be used to release
unused memory and allocate a single large block that will be able to satisfy
allocations up to the previously observed high-watermark. 

To call this
function, you must get the pointer to the
:class:`umpire::strategy::AllocationStrategy` from the
:class:`umpire::Allocator`:

.. literalinclude:: ../../../examples/cookbook/recipe_coalesce_pool.cpp
   :start-after: _sphinx_tag_tut_unwrap_strategy_start
   :end-before: _sphinx_tag_tut_unwrap_strategy_end
   :language: C++

Once you have the pointer to the appropriate strategy, you can call the
function:

.. literalinclude:: ../../../examples/cookbook/recipe_coalesce_pool.cpp
   :start-after: _sphinx_tag_tut_call_coalesce_start
   :end-before: _sphinx_tag_tut_call_coalesce_end
   :language: C++

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_coalesce_pool.cpp
