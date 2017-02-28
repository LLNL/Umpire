******
Umpire
******

Umpire is a resource management libray that allows the discovery, provision,
and management of memory on next-generation hardware architectures.


======
Design
======

The current Umpire design is made up of three main components:

-----------
MemorySpace
-----------

An interface defining how we can interact with a specific physical memory (e.g.
DRAM, GPU memory), it handles the details of a specific memory space, including
its properties and allocation/free.  Each MemorySpace maintains a map of
AllocationRecords, so that based on a given pointer we know how big it is.

----------------
Resource Manager
----------------

This handles the discovery and management of available MemorySpaces. It also
provides general methods for allocation and deletion of data, which can be
controlled with either a string or a specific MemorySpace. It also keeps track
of which space an allocation was made in. 


---------------
MemoryAllocator
---------------

MemoryAllocator is an interface that decouples memory allocation from the
specific MemorySpace. This allows us to provide different types of allocators
for a single physical memory space, for example, a memory pool.

---------------
MemoryOperation
---------------

Handles move/copy between two specific memory spaces. Need concrete
implementations to specialize what needs to happen.  These would be stored by
the ResourceManager, so that it can move data.  I think the operator registry
idea that SAMRAI uses would work well for this.  The final thing I was thinking
was having "AllocationSets" that group together allocations so that they can be
moved/deleted en masse. When I was thinking about this some more, it almost
seems like this maps somewhat to a communication-like infrastructure, where we
register allocations with a set, and then perform the operation all at once.
When coupled with Sidre, this would include even more of those concepts - we
could register variables with some AllocationAlgorithm, and then have specific
AllocationSchedules for an actual set of allocations (I think I've been
spending too much time looking at the SAMRAI source).
