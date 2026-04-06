.. _introspection:

=============
Introspection
=============

When writing code to run on computers with a complex memory hierarchy, one of
the most difficult things can be keeping track of where each pointer has been
allocated. Umpire's instrospection capability keeps track of this information,
as well as other useful bits and pieces you might want to know.

Umpire supports multiple *introspection levels* to control the overhead of
recording allocation metadata:

- ``off``: disable public introspection queries
- ``basic``: track only exact allocation-pointer ownership
- ``on``: track full allocation metadata and backtraces (if enabled via ``UMPIRE_BACKTRACE``)

The level can be set via the ``UMPIRE_INTROSPECTION_LEVEL`` environment
variable, or programmatically with
``umpire::ResourceManager::setIntrospectionLevel``.

The :class:`umpire::ResourceManager` can be used to find the allocator
associated with an address: 

.. literalinclude:: ../../../examples/tutorial/tut_introspection.cpp
   :start-after: _sphinx_tag_tut_getallocator_start
   :end-before: _sphinx_tag_tut_getallocator_end
   :language: C++

Once you have this, it's easy to query things like the name of the Allocator or
find out the associated :class:`umpire::Platform`, which can help
you decide where to operate on this data:

.. literalinclude:: ../../../examples/tutorial/tut_introspection.cpp
   :start-after: _sphinx_tag_tut_getinfo_start
   :end-before: _sphinx_tag_tut_getinfo_end
   :language: C++

You can also find out how big the allocation is, in case you forgot:

.. literalinclude:: ../../../examples/tutorial/tut_introspection.cpp
   :start-after: _sphinx_tag_tut_getsize_start
   :end-before: _sphinx_tag_tut_getsize_end
   :language: C++

These functions require the global introspection level to be ``on``. With
``basic``, only exact-pointer ownership checks such as
``umpire::ResourceManager::hasAllocator`` remain available.

.. literalinclude:: ../../../examples/tutorial/tut_introspection.cpp
