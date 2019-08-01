.. _introspection:

=============
Introspection
=============

When writing code to run on computers with a complex memory hierarchy, one of
the most difficult things can be keeping track of where each pointer has been
allocated. Umpire's instrospection capability keeps track of this information,
as well as other useful bits and pieces you might want to know.

The :class:`umpire::ResourceManager` can be used to find the allocator
associated with an address: 

.. literalinclude:: ../../../examples/tutorial/tut_introspection.cpp
                    :lines: 36

Once you have this, it's easy to query things like the name of the Allocator:

.. literalinclude:: ../../../examples/tutorial/tut_introspection.cpp
                    :lines: 39

You can also find out the associated :class:`umpire::Platform`, which can help
you decide where to operate on this data:

.. literalinclude:: ../../../examples/tutorial/tut_introspection.cpp
                    :lines: 41

You can also find out how big the allocation is, in case you forgot:

.. literalinclude:: ../../../examples/tutorial/tut_introspection.cpp
                    :lines: 44

Remember that these functions will work on any allocation made using an
Allocator or :class:`umpire::TypedAllocator`.

.. literalinclude:: ../../../examples/tutorial/tut_introspection.cpp
