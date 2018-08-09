.. _operations:

==========
Operations
==========

Moving and modifying data in a heterogenous memory system can be annoying. You
have to keep track of the source and destination, and often use vendor-specific
APIs to perform the modifications. In Umpire, all data modification and
movement is wrapped up in a concept we call `operations`. Full documentation
for all of these is available here: :namespace:`umpire::op`. The full code
listing for each example is include at the bottom of the page.

----
Copy
----

Let's start by looking at how we copy data around. The
:class:`umpire::ResourceManager` provides an interface to copy that handles
figuring out where the source and destination pointers were allocated, and
selects the correct implementation to copy the data:

.. literalinclude:: ../../../examples/tutorial/tut_copy.cpp
                    :lines: 23-26

This example allocates the destination data using any valid Allocator. 


----
Move
----

If you want to move data to a new Allocator and deallocate the old copy, Umpire
provides a `move` operation.

.. literalinclude:: ../../../examples/tutorial/tut_move.cpp
                    :lines: 25-26

The move operation combines an allocation, a copy, and a deallocate into one
function call, allowing you to move data without having to have the destination
data allocated. As always, this operation will work with any valid destination
Allocator.

------
Memset
------

Setting a whole block of memory to a value (like 0) is a common operation, that
most people know as a memset. Umpire provides a memset implementation that can
be applied to any allocation, regardless of where it came from:

.. literalinclude:: ../../../examples/tutorial/tut_memset.cpp
                    :lines: 22

----------
Reallocate
----------


-------------
Memory Advice
-------------



--------
Listings
--------

.. literalinclude:: ../../../examples/tutorial/tut_copy.cpp

.. literalinclude:: ../../../examples/tutorial/tut_move.cpp
