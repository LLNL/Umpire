.. _dynamic_pool:

=============
Dynamic Pools
=============

Frequently allocating and deallocating memory can be quite costly, especially
when you are making large allocations or allocating on different memory
resources. To mitigate this, Umpire provides allocation strategies that can be
used to customize how data is obtained from the system.

In this example, we will look at the :class:`umpire::strategy::DynamicPoolList`
strategy. This is a simple pooling algorithm that can fulfill requests for
allocations of any size. To create a new ``Allocator`` using the
:class:`umpire::strategy::DynamicPoolList` strategy:

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_1.cpp
   :start-after: _sphinx_tag_tut_makepool_start
   :end-before: _sphinx_tag_tut_makepool_end
   :language: C++

We have to provide a new name for the Allocator, as well as the underlying
Allocator we wish to use to grab memory.

Additionally, in the previous section on Allocators, we mentioned that you 
could build a new allocator off of an existing one using the ``getAllocator``
function. Here is another example of this, but using a strategy:

.. code-block:: bash

   umpire::Allocator addon_allocator = rm.makeAllocator<umpire::strategy::SizeLimiter>(
   resource + "_addon_pool", rm.getAllocator(pooled_allocator.getName()), 2098);
   
The purpose of this example is to show that the ``getAllocator`` function
can be used more than just to get an initial allocator. The ``addon_allocator`` will
be a dynamic pool allocator that is limited to 2098 bytes. Another good use case
for the ``getAllocator``  function is grabbing each available allocator in a loop and
querying some property. (Note that ``addon_allocator`` in the above example will be 
created with the same memory resource as ``pooled_allocator`` was.) 

Once you have an ``Allocator``, you can allocate and deallocate memory as
before, without needing to worry about the underlying algorithm used for the
allocations:

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_1.cpp
   :start-after: _sphinx_tag_tut_allocate_start
   :end-before: _sphinx_tag_tut_allocate_end
   :language: C++

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_1.cpp
   :start-after: _sphinx_tag_tut_deallocate_start
   :end-before: _sphinx_tag_tut_deallocate_end
   :language: C++

Don't forget, these strategies can be created on top of any valid Allocator:

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_1.cpp
   :start-after: _sphinx_tag_tut_anyallocator_start
   :end-before: _sphinx_tag_tut_anyallocator_end
   :language: C++

Most Umpire users will make allocations that use the GPU via the
:class:`umpire::strategy::DynamicPoolList`, to help mitigate the cost of allocating
memory on these devices.

You can tune the way that :class:`umpire::strategy::DynamicPoolList` allocates
memory using two parameters: the initial size, and the minimum size. The
initial size controls how large the first underly allocation made will be,
regardless of the requested size. The minimum size controls the minimum size of
any future underlying allocations. These two parameters can be passed when
constructing a pool:

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_2.cpp
   :start-after: _sphinx_tag_tut_allocator_tuning_start
   :end-before: _sphinx_tag_tut_allocator_tuning_end
   :language: C++

Depending on where you are allocating data, you might want to use different
sizes. It's easy to construct multiple pools with different configurations:

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_2.cpp
   :start-after: _sphinx_tag_tut_device_sized_pool_start
   :end-before: _sphinx_tag_tut_device_sized_pool_end
   :language: C++

There are lots of different strategies that you can use, we will look at some
of them in this tutorial. A complete list of strategies can be found `here
<../features/operations.html>`_.

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_1.cpp

.. literalinclude:: ../../../examples/tutorial/tut_dynamic_pool_2.cpp
