.. _allocator_accessibility:

=========
Allocator Accessibility
=========

The Umpire library provides a variety of :class:`umpire::resource::MemoryResource` s 
which can be used by :class:`umpire::Allocator` s depending on what's available on
your system. The resources are explained more on the `Resources <https://umpire.readthedocs.io/en/develop/tutorial/resources.html>`_
page.

Additionally, the `platforms <https://github.com/LLNL/Umpire/blob/develop/src/umpire/util/Platform.hpp>`_ that Umpire supports is defined by the CAMP library.
This means that there is also a selection of platforms for which an allocator can
be associated with as well. Because of these options, it can be difficult to trace
not only which memory resource an allocator has been created with but also
which allocators can be accessed by which platforms.

Umpire provides the ability to query which memory resource is associated with a
particular allocator (link example/test). Additionally, Umpire has a feature
that exposes which allocators are accessible by which platforms (link example/
test).

Allocator Accessibility Table
-----------------------------
The following is a truth table showing whether or not an allocator will be accessible.
For example, if :class:`umpire::Allocator` ``alloc`` is created with the host memory 
resource and I want to know if it should be accessible from the ``omp_target`` CAMP
platform, then I can look at the corresponding entry in the table and find that it 
should be accessible.

.. list-table:: Allocator Accessibility
   :header-rows: 1
   :stub-columns: 1

   * - 
     - Undefined
     - host
     - cuda
     - omp_target
     - hip
     - sycl
   * - Unknown
     - F
     - F
     - F
     - F
     - F
     - F
   * - host
     - F
     - T
     - T*
     - T
     - T*
     - F
   * - device
     - F
     - T*
     - T
     - T
     - T
     - T
   * - device_const
     - F
     - F
     - T
     - X
     - T
     - X
   * - um
     - F
     - T
     - T
     - X
     - T
     - T
   * - pinned
     - F
     - T
     - T
     - X
     - T
     - T
   * - file
     - F
     - T
     - F
     - F
     - F
     - F

.. note:: 
  In the table, ``T`` means **true**, ``F`` means **false**, ``*`` means **conditional**,
  and ``X`` means **does not exist**.
  
Allocator Inaccessibility Configuration
---------------------------------------

This is where I will show how to configure Umpire to also double check that those memory 
resources that should not be accessible given a specific platform are indeed not accessible.
(They result in an exception using the EXPECT_DEATH gtest function.)

Build and Run Configuration
---------------------------

The ``allocator_integration_tests.cpp`` file checks what platforms are available and confirms
that all memory resources which should be accessible to that platform can actually be 
accessed. On the other hand, the ``allocator_access.cpp`` file shows all available memory 
resource types for an allocator, given the *host* platform.

To build and run these files, use the appropriate cmake flags for the desired platform and then
run ``ctest -C test -R allocator_accessibility_tests --output-on-failure`` for the test code and
``./bin/allocator_access`` for the example code.

.. note::
   The `Developer's Guide <https://umpire.readthedocs.io/en/develop/developer_guide.html>`_ shows
   how to configure Umpire with uberenv to build with different CAMP platforms.

Below, the ``allocator_access.cpp`` code is shown to demonstrate how this functionality can be 
used during development.

CHANGE EXAMPLE LINK

.. literalinclude:: ../../../examples/allocator.c
