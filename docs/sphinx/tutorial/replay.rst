.. _replay:

======
Replay
======

Umpire provides a lightweight replay capability that can be used to investigate
performance of particular allocation patterns and reproduce bugs. By running an
executable that uses Umpire with the environment variable ``UMPIRE_REPLAY`` set
to ``On``, Umpire will print out one line for each of the following events:

- :func:`umpire::ResourceManager::makeAllocator`
- :func:`umpire::Allocator::allocate`
- :func:`umpire::Allocator::deallocate`

The log can be captured and stored as a CSV file, then used as input to the
``replay`` application (avaible under the ``bin`` directory). The ``replay``
program will read the replay log, and recreate the events that occured as part
of the run that generated the log.

The file ``tut_replay.cpp`` makes a :class:`umpire::strategy::DynamicPool`:

.. literalinclude:: ../../../examples/tutorial/tut_replay.cpp
                    :lines: 35-37

This allocator is used to perform some randomly sized allocations, and later
free them:

.. literalinclude:: ../../../examples/tutorial/tut_replay.cpp
                    :lines: 41

.. literalinclude:: ../../../examples/tutorial/tut_replay.cpp
                    :lines: 45

Running this program with the ``UMPIRE_REPLAY`` environment variable set to
``On`` will generate the log file found in `tut_replay_log.csv`. You can see
that this file contains lines for making the :class:`umpire::Allocator`:

.. literalinclude:: ../../../examples/tutorial/tut_replay_log.csv
                    :lines: 1

Doing allocations:

.. literalinclude:: ../../../examples/tutorial/tut_replay_log.csv
                    :lines: 4

and finally, doing the deallocations:

.. literalinclude:: ../../../examples/tutorial/tut_replay_log.csv
                    :lines: 80 

                 

Loading this file with the ``replay`` program will replay this sequence of
:class:`umpire::Allocator` creation, allocations, and deallocations:

.. code-block:: bash

   ./bin/replay -inputfile ../tutorial/examples/tut_replay_log.csv
