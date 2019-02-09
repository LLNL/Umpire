.. _numa_allocator:

===========================================
Create an allocator on a specific NUMA node
===========================================

When using NUMA (cache coherent or non uniform memory access) systems,
there are different latencies to parts of the memory. From an
application perspective, the memory looks the same, yet especially for
high-performance computing it is advantageous to have finer control.
`malloc()` attempts to allocate memory close to your node, but it can
make no guarantees. Therefore, Linux provides both a process-level
interface for setting NUMA policies with the system utility `numactl`,
and a fine-grained interface with `libnuma`.

The NUMA node ID is one component of
:class:`umpire::resource::MemoryResourceTraits`. An allocator for a
specific node can be created by making a traits object and passing it
to :meth:`umpire::ResourceManager::getAllocatorFor`. This will
allocate memory through libnuma on the specific node. Note that Umpire
NUMA calls are done through wrapper in the `op` library under the
`numa::` namespace.

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_numa_allocator.cpp
