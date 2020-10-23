.. _dynamic_pool_heuristics:

===========================================================
Improving DynamicPool Performance with a Coalesce Heuristic
===========================================================

As needed, the
:class:`umpire::strategy::DynamicPool`
will continue to allocate
blocks to satisfy allocation requests that cannot be satisfied by blocks
currently in the pool it is managing.  Under certain application-specific
memory allocation patterns, fragmentation within the blocks or allocations that
are for sizes greater than the size of the largest available block can cause
the pool to grow too large.  For example, a problematic allocation pattern is
when an application makes several allocations of incrementing size where each
allocation is larger than the previous block size allocated.

The
:func:`umpire::strategy::DynamicPool::coalesce`
method may be used to cause the 
:class:`umpire::strategy::DynamicPool`
to coalesce the releasable blocks into a single larger block.  This is
accomplished by: tallying the size of all blocks without allocations against
them, releasing those blocks back to the memory resource, and creating a new
block of the previously tallied size.

Applications may offer a heuristic function to the
:class:`umpire::strategy::DynamicPool`
during instantiation that will return true whenever a pool reaches a
specific threshold of releasable bytes (represented by completely free blocks)
to the total size of the pool.  The DynamicPool will call this heuristic
function just before it returns from its
:func:`umpire::strategy::DynamicPool::deallocate`
method and when the function returns true, the DynamicPool will call
the :func:`umpire::strategy::DynamicPool::coalesce`
method.

The default heuristic of 100 will cause the DynamicPool to automatically
coalesce when all of the bytes in the pool are releasable and there is more
than one block in the pool.

A heuristic of 0 will cause the DynamicPool to never automatically coalesce.

Creation of the heuristic function is accomplished by:

.. literalinclude:: ../../../examples/cookbook/recipe_dynamic_pool_heuristic.cpp
   :start-after: _sphinx_tag_tut_creat_heuristic_fun_start
   :end-before: _sphinx_tag_tut_creat_heuristic_fun_end
   :language: C++

The heuristic function is then provided as a parameter when the object is
instantiated:

.. literalinclude:: ../../../examples/cookbook/recipe_dynamic_pool_heuristic.cpp
   :start-after: _sphinx_tag_tut_use_heuristic_fun_start
   :end-before: _sphinx_tag_tut_use_heuristic_fun_end
   :language: C++

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_dynamic_pool_heuristic.cpp
