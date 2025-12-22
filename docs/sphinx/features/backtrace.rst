.. _backtrace:

=========
Backtrace
=========
The Umpire library may be configured to provide using programs with backtrace
information as part of Umpire thrown exception description strings.

Umpire may also be configured to collect and provide backtrace information for
each Umpire provided memory allocation performed.

Build Configuration
-------------------
Backtrace is enabled in Umpire builds with the following:

- ``cmake ... -DUMPIRE_ENABLE_BACKTRACE=On ...`` to backtrace capability in Umpire.
- ``cmake -DUMPIRE_ENABLE_BACKTRACE=On -DUMPIRE_ENABLE_BACKTRACE_SYMBOLS=On ...`` to
  enable Umpire to display symbol information with backtrace.  

.. note::
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
:func:`umpire::print_allocator_records` free function.

An example for checking and displaying the information this information
logged above may be found here:

.. literalinclude:: ../../../examples/backtrace_example.cpp

Header Introspection Mode Limitations
--------------------------------------

When Umpire is built with header-based introspection (``UMPIRE_ENABLE_HEADER_INTROSPECTION=On``),
allocation backtraces and leak reporting have important limitations:

**Limited Backtrace Support**:
  - Allocation backtraces are only stored for allocations registered in the ResourceManager's
    map (``m_allocations``)
  - Header-tracked allocations bypass map registration and therefore have no backtraces
  - Only manually registered allocations and untracked allocator allocations appear in
    backtrace reports

**Affected Functions**:
  - ``umpire::print_allocator_records(allocator)`` - Only prints map-registered allocations
  - ``umpire::get_allocator_records(allocator)`` - Only returns map-registered allocations
  - ``umpire::get_leaked_allocations(allocator)`` - Only detects map-registered leaks
  - ``umpire::get_backtrace(ptr)`` - Only works if ptr was registered in the map

**Recommendation**:
  For comprehensive leak detection and backtrace reporting, use map-based introspection
  (``UMPIRE_ENABLE_HEADER_INTROSPECTION=Off``). Header-based introspection prioritizes
  performance over debugging capabilities.

.. note::
   This is a fundamental architectural tradeoff: header-based introspection eliminates the
   global allocation map to improve performance, which also eliminates the central registry
   needed for leak enumeration and backtrace storage.
