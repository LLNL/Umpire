.. _dynamic_pool:

=============
Dynamic Pools
=============

Frequently allocating and deallocating memory can be quite costly, especially
when you are making large allocations or allocating on different memory
resources. To mitigate this, Umpire provides allocation strategies that can be
used to customize how data is obtained from the system.

In this example, we will look at the ``DynamicPool`` strategy. This is a simple
pooling algorithm that can fulfill requests for allocations of any size. To
create a new ``Allocator`` using the ``DynamicPool`` strategy:

.. literalinclude:: ../../../examples/tut_dynamic_pool.cpp
                    :lines: 26-28

We have to provide a new name for the Allocator, as well as the underlying
Allocator we wish to use to grab memory.

Once you have an ``Allocator``, you can allocate and deallocate memory as
before, without needing to worry about the underlying algorithm used for the
allocations:

.. literalinclude:: ../../../examples/tut_dynamic_pool.cpp
                    :lines: 30-36

Don't forget, these strategies can be created on top of any valid Allocator:

.. literalinclude:: ../../../examples/tut_dynamic_pool.cpp
                    :lines: 42-48

Most Umpire users will make alloctations that use the GPU via the DynamicPool,
to help mitigate the cost of allocating memory on these devices.

There are lots of different strategies that you can use, and we will look at
many of them in the rest of this tutorial. A complete list of strategies can be
found here.
