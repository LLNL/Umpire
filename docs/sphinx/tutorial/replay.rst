.. _replay:

======
Replay
======
Umpire provides a lightweight replay capability that can be used to investigate
performance of particular allocation patterns and reproduce bugs.

Input Example
-------------
When replay is enabled, Umpire captures replay events and writes them as
JSON-formatted lines into a ``.stats`` file. This file can be used as input to
the ``replay`` application (available under the ``bin`` directory), which will
recreate the events that occurred as part of the run that generated the log.

The file ``tut_replay.cpp`` makes a :class:`umpire::strategy::QuickPool`:

.. literalinclude:: ../../../examples/tutorial/tut_replay.cpp
   :start-after: _sphinx_tag_tut_replay_make_allocate_start
   :end-before: _sphinx_tag_tut_replay_make_allocate_end
   :language: C++

This allocator is used to perform some randomly sized allocations, and later
free them:

.. literalinclude:: ../../../examples/tutorial/tut_replay.cpp
   :start-after: _sphinx_tag_tut_replay_allocate_start
   :end-before: _sphinx_tag_tut_replay_allocate_end
   :language: C++

.. literalinclude:: ../../../examples/tutorial/tut_replay.cpp
   :start-after: _sphinx_tag_tut_replay_dealocate_start
   :end-before: _sphinx_tag_tut_replay_dealocate_end
   :language: C++

Running the Example
-------------------
Running this program:

.. code-block:: bash

   UMPIRE_REPLAY="On" ./bin/examples/tutorial/tut_replay

will write Umpire replay events to a file with a name like
``umpire.<pid>.<uid>.stats`` in the current directory (or in the directory
specified by ``UMPIRE_OUTPUT_DIR``). This file contains JSON formatted lines.

Replaying the session
---------------------
Loading this ``.stats`` file with the ``replay`` program will replay this
sequence of :class:`umpire::Allocator` creation, allocations, and
deallocations:

.. code-block:: bash

   ./bin/replay -i umpire.<pid>.<uid>.stats
