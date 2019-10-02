.. _allocators:

=======================
FORTRAN API: Allocators
=======================

The fundamental concept for accessing memory through Umpire is an
:class:`umpire:Allocator`. In FORTRAN, this means using the type
``UmpireAllocator``. This type provides an ``allocate_pointer`` function to
allocate raw memory, and a generic ``allocate`` procedure that takes an array
pointer and an array of dimensions and will allocate the correct amount of
memory.

As with the native C++ interface, all allocators are accessed via the
:class:`umpire::ResourceManager`. In the FORTRAN API, there is a corresponding
``UmpireResourceManager`` type. To get an ``UmpireAllocator``:

.. literalinclude:: ../../../../examples/tutorial/fortran/tut_allocator.f
                    :lines: 17-18

In this example we fetch the allocator by id, using 0 means you will always get
a host allocator. Once you have an ``UmpireAllocator``, you can use it to allocate and
deallocate memory:

.. literalinclude:: ../../../../examples/tutorial/fortran/tut_allocator.f
                    :lines: 20-24

In this case, we allocate a one-dimensional array using the generic
``allocate`` function.
