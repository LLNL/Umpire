.. _replay:

======
Replay
======
Umpire provides a lightweight replay capability that can be used to investigate
performance of particular allocation patterns and reproduce bugs.

Input Example
-------------
A log can be captured and stored as a JSON file, then used as input to the
``replay`` application (available under the ``bin`` directory). The ``replay``
program will read the replay log, and recreate the events that occured as part
of the run that generated the log.

The file ``tut_replay.cpp`` makes a :class:`umpire::strategy::DynamicPool`:

.. literalinclude:: ../../../examples/tutorial/tut_replay.cpp
                    :lines: 27-29
                    :language: c++

This allocator is used to perform some randomly sized allocations, and later
free them:

.. literalinclude:: ../../../examples/tutorial/tut_replay.cpp
                    :lines: 32-33

.. literalinclude:: ../../../examples/tutorial/tut_replay.cpp
                    :lines: 36

Running the Example
-------------------
Running this program:

.. code-block:: bash

   UMPIRE_REPLAY="On" ./bin/examples/tutorial/tut_replay > tut_replay_log.json

will write Umpire replay events to the file ``tut_replay_log.json``. You can
see that this file contains JSON formatted lines.

Replaying the session
---------------------
Loading this file with the ``replay`` program will replay this sequence of
:class:`umpire::Allocator` creation, allocations, and deallocations:

.. code-block:: bash

   ./bin/replay -i ../tutorial/examples/tut_replay_log.json
