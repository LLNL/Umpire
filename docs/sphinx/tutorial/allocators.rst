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
   :start-after: _sphinx_tag_tut_get_allocator_start
   :end-before: _sphinx_tag_tut_get_allocator_end
   :language: C++

You can also use an existing allocator to build another allocator off of it:

.. literalinclude:: ../../../examples/tutorial/tut_allocator.cpp
   :start-after: _sphinx_tag_tut_getAllocator_start
   :end-before: _sphinx_tag_tut_getAllocator_end
   :language: C++

This "add-on" allocator will also be built with the same memory resource. More on memory 
resources in the next section. Additionally, once you have an :class:`umpire::Allocator`
you can use it to allocate and deallocate memory:

.. literalinclude:: ../../../examples/tutorial/tut_allocator.cpp
   :start-after: _sphinx_tag_tut_allocate_start
   :end-before: _sphinx_tag_tut_allocate_end
   :language: C++

.. literalinclude:: ../../../examples/tutorial/tut_allocator.cpp
   :start-after: _sphinx_tag_tut_deallocate_start
   :end-before: _sphinx_tag_tut_deallocate_end
   :language: C++

In the next section, we will see how to allocate memory using different
resources.

.. literalinclude:: ../../../examples/tutorial/tut_allocator.cpp
   :language: C++
