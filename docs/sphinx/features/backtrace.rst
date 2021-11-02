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
