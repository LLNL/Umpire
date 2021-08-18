.. _allocator_accessibility:

=========
Allocator Accessibility
=========

The Umpire library provides a variety of :class:`umpire::resource::MemoryResource` s 
which can be used to create :class:`umpire::Allocator` s depending on what's available on
your system. The resources are explained more on the `Resources <https://umpire.readthedocs.io/en/develop/tutorial/resources.html>`_
page.

Additionally, the `platforms <https://github.com/LLNL/Umpire/blob/develop/src/umpire/util/Platform.hpp>`_ that Umpire supports is defined by the `CAMP <https://github.com/LLNL/camp/blob/master/include/camp/resource/platform.hpp>`_ library.
This means that there is also a selection of platforms for which an allocator can
be associated with as well. For example, an Allocator created with the pinned memory
resource can be used with the host, cuda, hip, or sycl platforms.

Because of these options, it can be difficult to trace
not only which memory resource an allocator has been created with but also
which allocators can be accessed by which platforms. Umpire has the memory resource
trait, ``resource_type``, to provide the ability to query which memory resource is 
associated with a particular allocator (See example `here <https://github.com/LLNL/Umpire/blob/develop/tests/integration/memory_resource_traits_tests.cpp>`_). 

Additionally, Umpire has a function, ``is_accessible(Platform p, Allocator a)``, that determines 
if a particular allocator is accessible by a particular platform 
(See example `here <https://github.com/LLNL/Umpire/blob/develop/tests/integration/allocator_accessibility.cpp>`_). The ``allocator_accessibility.cpp`` test checks what platforms are available and confirms that all memory resources which should be accessible 
to that platform can actually be accessed and used.

For example, if a :class:`umpire::Allocator`, ``alloc``, 
is created with the host memory resource and we want to know if it should be 
accessible from the ``omp_target`` CAMP platform, then we can use the ``is_accessible(Platform::omp_target, alloc)`` 
function and find that it should be accessible. The ``allocator_access.cpp`` file demonstrates this
functionality for the *host* platform specifically.
  
Allocator Inaccessibility Configuration
---------------------------------------

On a different note, for those allocators that are deemed inaccessible, it may be useful to 
double check or confirm that the allocator can in fact NOT access memory on that given platform. 
In this case, the cmake flag, ``ENABLE_INACCESSIBILITY_TESTS``, will need to be turned on.

Build and Run Configuration
---------------------------

To build and run these files, either use uberenv or the appropriate cmake flags for the 
desired platform and then run ``ctest -T test -R allocator_accessibility_tests --output-on-failure`` 
for the test code and ``./bin/alloc_access`` for the example code.

.. note::
   The `Developer's Guide <https://umpire.readthedocs.io/en/develop/developer/uberenv.html>`_ shows
   how to configure Umpire with uberenv to build with different CAMP platforms.

Below, the ``allocator_access.cpp`` code is shown to demonstrate how this functionality can be 
used during development.

.. literalinclude:: ../../../examples/allocator_access.cpp
