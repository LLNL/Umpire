.. _allocators:

==========
Allocators
==========

The fundamental concept for accessing memory through Umpire is the
:class:`umpire::Allocator`. An :class:`umpire::Allocator` is a C++ object that
can be used to allocate and deallocate memory, as well as query a pointer to
get some extra information about it.

All :class:`umpire::Allocator` s are created and managed by Umpire's
:class:`umpire::ResourceManager`. To get an Allocator, you need to ask for one: 

.. literalinclude:: ../../../examples/tutorial/tut_allocator.cpp
                    :lines: 22-24

Once you have an :class:`umpire::Allocator` you can use it to allocate and deallocate memory:

.. literalinclude:: ../../../examples/tutorial/tut_allocator.cpp
                    :lines: 26-32

In the next section, we will see how to allocate memory using different
resources.

.. literalinclude:: ../../../examples/tutorial/tut_allocator.cpp
