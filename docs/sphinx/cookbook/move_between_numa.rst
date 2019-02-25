.. _move_between_numa::

==============================
Move memory between NUMA nodes
==============================

 When using NUMA (cache coherent or non uniform memory access) systems, there
are different latencies to parts of the memory. From an application
perspective, the memory looks the same, yet especially for high-performance
computing it is advantageous to have finer control. `malloc()` attempts to
allocate memory close to your node, but it can make no guarantees. Therefore,
Linux provides both a process-level interface for setting NUMA policies with
the system utility `numactl`, and a fine-grained interface with `libnuma`.
These interfaces work on ranges of memory in multiples of the page size, which
is the length or unit of address space loaded into a processor cache at once.

A page range may be bound to a NUMA node using the
:class:`umpire::strategy::NumaPolicy`. It can therefore also be moved between
NUMA nodes using the :meth:`umpire::ResourceManager::move` with a different
allocator. The power of using such an abstraction is that the NUMA node can be
associated with a device, in which case the memory is moved to, for example,
GPU memory.

In this recipe we create an allocation bound to a NUMA node, and move it to
another NUMA node.

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_move_between_numa.cpp
