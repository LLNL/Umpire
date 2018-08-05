.. _resources:

=========
Resources
=========

Each computer system will have a number of distinct places in which the system
will allow you to allocate memory. In Umpire's world, these are
``MemoryResources``. A memory resource can correspond to a hardware resource,
but can also be used to identify memory with a particular characteristic, like
"pinned" memory in a GPU system.

When you configure Umpire, it will create ``MemoryResource`` s according to
what is available on the system you are building for. For each resource, Umpire
will create a default ``Allocator`` that you can use. In the previous example,
we were actually using an ``Allocator`` created for the ``MemoryResource``
corresponding to the CPU memory 

The easiest way to identify resources is by name. The "HOST" resource is always
available. In a modern NVIDIA GPU system, we also have resources that represent
global GPU memory ("DEVICE"), unified memory that can be accessed by the CPU or
GPU ("UM") and host memory that can be accessed by the GPU ("PINNED");

Umpire will create an ``Allocator`` for each of these resources, and you can
get them using the same ``getAllocator`` call you saw in the previous example:

.. literalinclude:: ../../../examples/tut_resources.cpp 
                    :lines 22-24

Note that every ``Allocator`` supports the same calls, no matter which resource
it is for, this means we can parametertize the ``getAllocator`` call, and run
the same code for all the resources available in the system:

.. literalinclude:: ../../../examples/tut_resources.cpp 
                    :lines 18-35

As you can see, we can call this function with any valid resource name:

.. literalinclude:: ../../../examples/tut_resources.cpp 
                    :lines 38-44

In the next example, we will learn how to move data between resources using
operations.
