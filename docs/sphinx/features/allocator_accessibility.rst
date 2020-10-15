.. _allocator_accessibility:

=========
Allocator Accessibility
=========

The Umpire library provides a variety of :class:`umpire::resource::MemoryResource`s 
which can be used by :class:`umpire::Allocator`s depending on what's available on
your system. The resources are explained more on the `Resources <https://umpire.readthedocs.io/en/develop/tutorial/resources.html>`_
page.

Because of the wide selection of memory resources, it can get difficult to trace
which memory resources can be accessed by which platforms. Blah, I need an outline...

-Allocators have memory resources that they're associated with by definition.
   -- link to resource info
-Allocators can be accessed by only certain platforms depending on the memory
resource it was created with.
   -- link to platform info
-This leads to potential confusion of which memory resource a particuar allocator
may be associated with (hence the ability to query a memory resource) but also 
confusion as to which allocators can be accessed by which platforms (hence ability
to figure out which allocators are accessible by which platforms).
-Allocator Accessibility table
-Example code + tests that are provided

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
