.. _typed_allocators:

----------------
Typed Allocators
----------------

Sometimes, you might want to construct an allocator that allocates objects of a
specific type. Umpire provides a :class:`umpire::TypedAllocator` for this
purpose. It can also be used with STL objects like ``std::vector``.

A :class:`umpire::TypedAllocator` is constructed from any existing Allocator,
and provides the same interface as the normal :class:`umpire::Allocator`.
However, when you call allocate, this argument is the number of objects you
want to allocate, no the total number of bytes:

.. literalinclude:: ../../../examples/tutorial/tut_typed_allocator.cpp
                    :lines: 25-29

To use this allocator with an STL object like a vector, you need to pass the
type as a template parameter for the vector, and also pass the allocator to the
vector when you construct it:

.. literalinclude:: ../../../examples/tutorial/tut_typed_allocator.cpp
                    :lines: 32-33

One thing to remember is that whatever allocator you use with an STL object, it
must be compatible with the inner workings of that object. For example, if you
try and use a "DEVICE"-based allocator it will fail, since the vector will try
and construct each element. The CPU cannot access DEVICE memory in most
systems, thus causing a segfault. Be careful!

.. literalinclude:: ../../../examples/tutorial/tut_typed_allocator.cpp
