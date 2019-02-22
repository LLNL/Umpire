.. _pools:

============
C API: Pools
============

Frequently allocating and deallocating memory can be quite costly, especially
when you are making large allocations or allocating on different memory
resources. To mitigate this, Umpire provides allocation strategies that can be
used to customize how data is obtained from the system.

In this example, we will look at creating a pool that can fulfill requests for
allocations of any size. To create a new ``umpire_allocator`` using the pooling
algorithm:

.. literalinclude:: ../../../examples/tutorial/tut_pool.c
                    :lines: 24-28

The two arguments are the size of teh initial block that is allocated, and the
minimum size of any future blocks. We have to provide a new name for the
allocator, as well as the underlying ``umpire_allocator`` we wish to use to
grab memory.

Once you have the allocator, you can allocate and deallocate memory as
before, without needing to worry about the underlying algorithm used for the
allocations:

.. literalinclude:: ../../../examples/tutorial/c/tut_pool.c
                    :lines: 30-35


This pool can be created with any valid underlying ``umpire_allocator``.
