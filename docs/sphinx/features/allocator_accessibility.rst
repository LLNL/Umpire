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
For example, if ``:class:`umpire::Allocator` alloc`` is created with the host memory 
resource and I want to know if it should be accessible from the ``omp_target`` CAMP
platform, then I can look at the corresponding entry in the table and find that it 
should be accessible.

.. list-table:: Allocator Accessibility
   :header-rows: 2

   * - Undefined
     - host
     - cuda
     - omp_target
     - hip
     - sycl
   * - Unknown
     - host
     - device
     - device_const
     - um
     - pinned
     - file 
   * - F
     - F
     - F
     - F
     - F
     - F
   * - F
     - T
     - T*
     - T
     - T*
     - F
   * - F
     - T*
     - T
     - T
     - T
     - T
   * - F
     - F
     - T
     - X
     - T
     - X
   * - F
     - T
     - T
     - X
     - T
     - T
   * - F
     - T
     - T
     - X
     - T
     - T
   * - F
     - T
     - F
     - F
     - F
     - F

.. note:: In the table, ``T`` means ``true``, ``F`` means ``false``, ``*`` means ``conditional``,
  and ``X`` means ``does not exist``.

Build Configuration
-------------------
Backtrace is enabled in Umpire builds with the following:

- **``cmake ... -DENABLE_BACKTRACE=On ...``** to backtrace capability in Umpire.
- **``cmake -DENABLE_BACKTRACE=On -DENABLE_BACKTRACE_SYMBOLS=On ...``** to
  enable Umpire to display symbol information with backtrace.  **Note:**
  Using programs will need to add the ``-rdyanmic`` and ``-ldl`` linker flags
  in order to properly link with this configuration of the Umpire library.

Runtime Configuration
---------------------
For versions of the Umpire library that are backtrace enabled (from flags
above), the user may expect the following.

Backtrace information will always be provided in the description strings of
umpire generated exception throws.

Setting the environment variable ``UMPIRE_BACKTRACE=On`` will cause
Umpire to record backtrace information for each memory allocation it provides.

Setting the environment variable ``UMPIRE_LOG_LEVEL=Error`` will cause to
Umpire to log backtrace information for each of the leaked Umpire allocations
found during application exit.

A programatic interface is also availble via the
func::`umpire::print_allocator_records` free function.

An example for checking and displaying the information this information
logged above may be found here:

.. literalinclude:: ../../../examples/backtrace_example.cpp
