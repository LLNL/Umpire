.. _dynamic_pool_heuristics:

=======================================
Improving DynamicPool Performance with a Coalesce Heuristic
=======================================

By default, a DynamicPool object will not coalesce blocks that it allocates
for the pool.  As needed, the :class:`umpire::strategy::DynamicPool` will continue to allocate blocks to
satisfy allocation requests that cannot be satisfied by blocks currently in the
pool.  Under certain application specific memory allocation patterns, this can
lead to fragmentation within the pool which could cause the pool to grow too
large.  For example, a problematic allocation pattern is when an application
makes several allocations of incrementing size where each allocation is larger
than the previous block size allocated.

To help address this, applications may offer a heuristic function to the
DynamicPool object during instantiation that will return true whenever a pool
reaches a specific threshold of releasable bytes (represented by completely
free blocks) to the total size of the pool.  The DynamicPool will call this
heuristic function just before it returns from its deallocate() method and when
the function returns true, the DynamicPool will coalesce all of its releasable
blocks into a single larger block of the combined size.

A heuristic of 0 will cause the DynamicPool to never automatically coalesce.

A heuristic of 100 will cause the DynamicPool to automatically coalesce when
all of the bytes in the pool are releasable and there is more than one block
in the pool.

Creation of the heuristic function is accomplished by:

.. literalinclude:: ../../../examples/cookbook/recipe_dynamic_pool_heuristic.cpp
                    :lines: 30-35

The heuristic function is then provided as a parameter when the object is
instantiated:

.. literalinclude:: ../../../examples/cookbook/recipe_dynamic_pool_heuristic.cpp
                    :lines: 36-47

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_dynamic_pool_heuristic.cpp
