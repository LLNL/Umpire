.. _dynamic_pool:

=============
Dynamic Pools
=============

Frequently allocating and deallocating memory can be quite costly, especially
when you are making large allocations or allocating on different memory
resources. To mitigate this, Umpire provides allocation strategies that can be
used to customize how data is obtained from the system.

In this example, we will look at the :class:`umpire::strategy::DynamicPool`
strategy. This is a simple pooling algorithm that can fulfill requests for
allocations of any size. To create a new ``Allocator`` using the
:class:`umpire::strategy::DynamicPool` strategy:

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_1.cpp
                    :lines: 26-30

We have to provide a new name for the Allocator, as well as the underlying
Allocator we wish to use to grab memory.

Once you have an ``Allocator``, you can allocate and deallocate memory as
before, without needing to worry about the underlying algorithm used for the
allocations:

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_1.cpp
                    :lines: 32-38

Don't forget, these strategies can be created on top of any valid Allocator:

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_1.cpp
                    :lines: 46-50

Most Umpire users will make alloctations that use the GPU via the
:class:`umpire::strategy::DynamicPool`, to help mitigate the cost of allocating
memory on these devices.

You can tune the way that :class:`umpire::strategy::DynamicPool` allocates
memory using two parameters: the initial size, and the minimum size. The
initial size controls how large the first underly allocation made will be,
regardless of the requested size. The minimum size controls the minimum size of
any future underlying allocations. These two parameters can be passed when
constructing a pool:

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_2.cpp
                    :lines: 31-35

Depending on where you are allocating data, you might want to use different
sizes. It's easy to construct multiple pools with different configurations:

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_2.cpp
                    :lines: 49-54

There are lots of different strategies that you can use, we will look at some
of them in this tutorial. A complete list of strategies can be found `here
<../features/operations.html>`_.

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_1.cpp

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_2.cpp
