.. _tutorial:

===============
Umpire Tutorial
===============

This section is a tutorial introduction to Umpire. We start with the most basic
memory allocation, and move through topics like allocating on different
resources, using allocation strategies to change how memory is allocated, using
operations to move and modify data, and how to use Umpire introspection
capability to find out information about Allocators and allocations.

These examples are all built as part of Umpire, and you can find the files in
the `examples <https://github.com/LLNL/Umpire/tree/develop/examples>`_
directory at the root of the Umpire repository.  Feel free to play around and
modify these examples to experiment with all of Umpire's functionality.

The following tutorial examples assume a working knowledge of C++ and a general
understanding of how memory is laid out in modern heterogeneous computers. The
main thing to remember is that in many systems, memory on other execution
devices (like GPUs) might not be directly accessible from the CPU. If you try
and access this memory your program will error! Luckily, Umpire makes it easy
to move data around, and check where it is, as you will see in the following
sections.

.. toctree::
   :maxdepth: 1
   :caption: Tutorial

   tutorial/allocators.rst
   tutorial/resources.rst
   tutorial/operations.rst
   tutorial/dynamic_pool.rst
   tutorial/introspection.rst
   tutorial/typed_allocators.rst
   tutorial/replay.rst

We also have a tutorial for the C interface to Umpire. Complete example
listings are available, and will be compiled if you have configured Umpire with
``-DENABLE_C_API=On``. 

The C tutorial assumes an understanding of C, and it would be useful to have
some knowledge of C++ to understand how the C API maps to the native C++
classes that Umpire provides.

.. toctree::
   :maxdepth: 2
   :caption: C API Tutorial

   tutorial/c/allocators.rst
   tutorial/c/resources.rst
