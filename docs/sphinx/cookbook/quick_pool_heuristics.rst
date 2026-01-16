.. _quick_pool_heuristics:

===========================================================
Improving QuickPool Performance with a Coalesce Heuristic
===========================================================

As needed, the QuickPool memory pool (:class:`umpire::strategy::QuickPool`)
will continue to allocate
blocks to satisfy allocation requests that cannot be satisfied by blocks
currently in the pool it is managing.  Under certain application-specific
memory allocation patterns, fragmentation within the blocks or allocations that
are for sizes greater than the size of the largest available block can cause
the pool to grow too large.  For example, a problematic allocation pattern is
when an application makes several allocations of incrementing size where each
allocation is larger than the previous block size allocated.

The Umpire library provides **Coalescing Heuristics** to help manage the blocks
of memory within a memory pool. The purpose of a coalescing heuristic is to
ensure that the memory pool is properly maintained through the duration of
an application so that it does not grow too large and prematurely run out
of memory. There are two coalescing heuristics: **Percent Releasable** and 
**Blocks Releasable**. By default, a memory pool will use the Percent Releasable 
heuristic. Additionally, Umpire
provides an optional tuning parameter. This tuning is called the `HighWatermark` tuning and
it uses the pool's HighWatermark value when coalescing the
pool. By default, coalescing heuristics **DO NOT** use the `HighWatermark` tuning.
Instead, they use the pool's Actual Size value. Go to Umpire's RZ Confluence page under
"Design Documents" and refer to the "Umpire Pool Usage and Control" documentation for
more information. You can also refer to `Coalescing Pool Memory <https://umpire.readthedocs.io/en/develop/sphinx/cookbook/coalesce_pool.html>`_.

.. note::
      To turn on the HighWatermark heuristic tuning, use the coalescing
      heuristic with the "_hwm" suffix (e.g. :func:`umpire::strategy::QuickPool::blocks_releasable_hwm(num_blocks)`).

The
:func:`umpire::strategy::QuickPool::coalesce`
method may be used to cause the 
:class:`umpire::strategy::QuickPool`
to coalesce the releasable blocks into a single larger block.  This is
accomplished by: tallying the size of all blocks without allocations against
them, releasing those blocks back to the memory resource, and creating a new
block of the previously tallied size.

.. note::
      If the HighWatermark heuristic tuning is used, then the new block
      will be allocated to the pool's HighWatermark instead.

Applications may offer a heuristic function to the
:class:`umpire::strategy::QuickPool`
during instantiation that will return true whenever a pool reaches a
specific threshold of releasable bytes (represented by completely free blocks)
to the total size of the pool.  The QuickPool will call this heuristic
function just before it returns from its
:func:`umpire::strategy::QuickPool::deallocate`
method and when the function returns true, the QuickPool will call
the :func:`umpire::strategy::QuickPool::coalesce`
method.

The default heuristic of 100 will cause the QuickPool to automatically
coalesce when all of the bytes in the pool are releasable and there is more
than one block in the pool.

A heuristic of 0 will cause the QuickPool to never automatically coalesce.

Creation of the heuristic function is accomplished by:

.. literalinclude:: ../../../examples/cookbook/recipe_quick_pool_heuristic.cpp
   :start-after: _sphinx_tag_tut_creat_heuristic_fun_start
   :end-before: _sphinx_tag_tut_creat_heuristic_fun_end
   :language: C++

In this example, we are using the `Percent Releasable` heuristic. The heuristic 
function is then provided as a parameter when the object is instantiated:

.. literalinclude:: ../../../examples/cookbook/recipe_quick_pool_heuristic.cpp
   :start-after: _sphinx_tag_tut_use_heuristic_fun_start
   :end-before: _sphinx_tag_tut_use_heuristic_fun_end
   :language: C++

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_quick_pool_heuristic.cpp

