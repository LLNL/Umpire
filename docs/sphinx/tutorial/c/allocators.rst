.. _allocators:

=================
C API: Allocators
=================

The fundamental concept for accessing memory through Umpire is an
:class:`umpire:Allocator`. In C, this means using the type
``umpire_allocator``. There are corresponding functions that take an
``umpire_allocator`` and let you allocate and deallocate memory.

As with the native C++ interface, all allocators are accessed via the
:class:`umpire::ResourceManager`. In the C API, there is a corresponding
``umpire_resourcemanager`` type. To get an ``umpire_allocator``:

.. literalinclude:: ../../../../examples/tutorial/c/tut_allocator.c
   :start-after: _umpire_tut_c_get_allocator_start
   :end-before: _umpire_tut_c_get_allocator_end
   :language: C

Once you have an ``umpire_allocator``, you can use it to allocate and
deallocate memory:

.. literalinclude:: ../../../../examples/tutorial/c/tut_allocator.c
   :start-after: _umpire_tut_c_allocate_start
   :end-before: _umpire_tut_c_allocate_end
   :language: C

In the next section, we will see how to allocate memory in different places.
