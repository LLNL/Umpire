.. _shrinking_pools:

============================
Growing and Shrinking a Pool
============================

When sharing a pool between different parts of your application, or even
between co-ordinating libraries in the same application, you might want to grow
and shrink a pool on demand. By limiting the size of a pool using device
memory, you leave more space on the GPU for "unified memory" to move data there.

The basic idea is to create a pool that allocates a block of your minimum size,
and then allocate a single word from this pool to ensure the initial block is
never freed:

.. literalinclude:: ../../../examples/cookbook/recipe_shrink.cpp
                    :lines: 32-35

To increase the pool size you can preallocate a large chunk and then
immediately free it. The pool will retain this memory for use by later
allocations:

.. literalinclude:: ../../../examples/cookbook/recipe_shrink.cpp
                    :lines: 48-49

Assuming that there are no allocations left in the larger "chunk" of the pool,
you can shrink the pool back down to the initial size by calling
:func:`umpire::Allocator::release`:

.. literalinclude:: ../../../examples/cookbook/recipe_shrink.cpp
                    :lines: 58

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_shrink.cpp
