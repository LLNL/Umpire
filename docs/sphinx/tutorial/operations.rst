.. _operations:

==========
Operations
==========

Moving and modifying data in a heterogenous memory system can be annoying. You
have to keep track of the source and destination, and often use vendor-specific
APIs to perform the modifications. In Umpire, all data modification and
movement is wrapped up in a concept we call `Operations`.

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

.. literalinclude:: ../../../examples/tutorial/tut_copy.cpp

----
Move
----

If you want to move data to a new Allocator and deallocate the old copy, Umpire
provides a `move` operation.

.. literalinclude:: ../../../examples/tutorial/tut_move.cpp
                    :lines: 23-26

.. literalinclude:: ../../../examples/tutorial/tut_move.cpp

------
Memset
------



----------
Reallocate
----------



-------------
Memory Advice
-------------
