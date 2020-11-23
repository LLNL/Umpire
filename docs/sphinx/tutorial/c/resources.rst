.. _resources:

================
C API: Resources
================

Each computer system will have a number of distinct places in which the system
will allow you to allocate memory. In Umpire's world, these are `memory
resources`. A memory resource can correspond to a hardware resource, but can
also be used to identify memory with a particular characteristic, like "pinned"
memory in a GPU system.

When you configure Umpire, it will create
:class:`umpire::resource::MemoryResource` s according to what is available on
the system you are building for. For each resource, Umpire will create a
default ``umpire_allocator`` that you can use. In the previous example, we were
actually using an ``umpire_allocator`` created for the memory resource
corresponding to the CPU memory.

The easiest way to identify resources is by name. The "HOST" resource is always
available. In a system configured with NVIDIA GPUs, we also have resources that represent
global GPU memory ("DEVICE"), unified memory that can be accessed by the CPU or
GPU ("UM") and host memory that can be accessed by the GPU ("PINNED");

Umpire will create an ``umpire_allocator`` for each of these resources, and you
can get them using the same ``umpire_resourcemanager_get_allocator_by_name``
call you saw in the previous example:

.. literalinclude:: ../../../../examples/tutorial/c/tut_resources.cpp
   :start-after: _sphinx_tag_tut_create_allocator_start
   :end-before: _sphinx_tag_tut_create_allocator_end
   :language: C

Note that every allocator supports the same calls, no matter which resource it
is for, this means we can run the same code for all the resources available in
the system:

.. literalinclude:: ../../../../examples/tutorial/c/tut_resources.cpp 
   :start-after: _sphinx_tag_tut_allocate_start
   :end-before: _sphinx_tag_tut_allocate_end
   :language: C

As you can see, we can call this function with any valid resource name:

.. literalinclude:: ../../../../examples/tutorial/c/tut_resources.cpp 
   :start-after: _sphinx_tag_tut_resource_types_start
   :end-before: _sphinx_tag_tut_resource_types_end
   :language: C

In the next example, we will learn how to move data between resources using
operations.
