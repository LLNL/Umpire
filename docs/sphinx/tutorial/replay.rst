.. _replay:

======
Replay
======

Umpire provides a lightweight replay capability that can be used to investigate
performance of particular allocation patterns and reproduce bugs. By running an
executable that uses Umpire with the environment variable ``UMPIRE_REPLAY`` set
to ``On``, Umpire will emit information for the following Umpire events:

- **version**
  :func:`umpire::get_major_version`,
  :func:`umpire::get_minor_version`,
  and :func:`umpire::get_patch_version`
- **makeMemoryResource**
  :func:`umpire::resource::MemoryResourceRegistry::makeMemoryResource`
- **makeAllocator** :func:`umpire::ResourceManager::makeAllocator`
- **allocate** :func:`umpire::Allocator::allocate`
- **deallocate** :func:`umpire::Allocator::deallocate`

Input Example
-------------
The log can be captured and stored as a JSON file, then used as input to the
``replay`` application (avaible under the ``bin`` directory). The ``replay``
program will read the replay log, and recreate the events that occured as part
of the run that generated the log.

The file ``tut_replay.cpp`` makes a :class:`umpire::strategy::DynamicPool`:

.. literalinclude:: ../../../examples/tutorial/tut_replay.cpp
                    :lines: 35-37
                    :language: c++

This allocator is used to perform some randomly sized allocations, and later
free them:

.. literalinclude:: ../../../examples/tutorial/tut_replay.cpp
                    :lines: 41

.. literalinclude:: ../../../examples/tutorial/tut_replay.cpp
                    :lines: 45

Running the Example
-------------------
Running this program:

.. code-block:: bash

   UMPIRE_REPLAY="On" ./bin/examples/tutorial/tut_replay > tut_replay_log.json

will write Umpire replay events to the file ``tut_replay_log.json``. You can
see that this file contains JSON formatted lines.

Interpretting Results - Version Event
-------------------------------------
The first event captured is the **version** event which shows the version
information as follows:

.. literalinclude:: ../../../examples/tutorial/tut_replay_log.json
                    :lines: 15
                    :language: json

Each line contains the following set of common elements:

- **kind** - Always set to *replay*
- **uid** - This is the MPI rank of the process generating the event for mpi
            programs or the PID for non-mpi.
- **timestamp** - Set to the time when the event occurred
- **event** - Set to one of: *version*, *makeMemoryResource*, *makeAllocator*,
              *allocate*, or *deallocate*
- **payload** - Optional and varies upon event type
- **result** - Optional and varies upon event type

As can be seen, the *major*, *minor*, and *patch* version numbers are captured
within the *result* for this event.

makeMemoryResource Event
------------------------
Next you will see events for the creation of the default memory resources
provided by Umpire with the **makeMemoryResource** event:

.. literalinclude:: ../../../examples/tutorial/tut_replay_log.json
                    :lines: 16-20
                    :language: json

The *payload* shows that a memory resource was created for *HOST*, *DEVICE*,
*PINNED*, *UM*, and *DEVICE_CONST* respectively.  The *result* is a reference
to the object that was created within Umpire for that resource.

makeAllocator Event
-------------------
The **makeAllocator** event occurs whenever a new allocator instance is being
created.  Each call to *makeAllocator* will generate a pair of JSON lines.  The
first line will show the intent of the call and the second line will show both
the intent and the result.  This is because the makeAllocator call can fail
and keeping both the intent and result allows us to reproduce this failure
later.

:class:`umpire::Allocator`:

.. literalinclude:: ../../../examples/tutorial/tut_replay_log.json
                    :lines: 21-22
                    :language: json

The *payload* shows how the allocator was constructed.  The *result* shows the
reference to the allocated object.

allocate Event
--------------
Like the **makeAllocator** event, the **allocate** event is captured as an
intention/result pair so that an error may be replayed in the event that
there is an allocation failure.

.. literalinclude:: ../../../examples/tutorial/tut_replay_log.json
                    :lines: 23-24
                    :language: json

The *payload* shows the object reference of the allocator and the size of the
allocation request.  The *result* shows the pointer to the memory allocated.

deallocate Event
----------------
.. literalinclude:: ../../../examples/tutorial/tut_replay_log.json
                    :lines: 151
                    :language: json

The *payload* shows the reference to the allocator object and the pointer
to the allocated memory that is to be freed.

Replaying the session
---------------------
Loading this file with the ``replay`` program will replay this sequence of
:class:`umpire::Allocator` creation, allocations, and deallocations:

.. code-block:: bash

   ./bin/replay -i ../tutorial/examples/tut_replay_log.json
