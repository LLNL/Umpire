.. _move_to_managed:

==================================
Moving Host Data to Managed Memory
==================================

When using a system with NVIDIA GPUs, you may realize that some host data
should be moved to unified memory in order to make it accessible by the GPU.
You can do this with the :func:`umpire::ResourceManager::move` operation:

.. literalinclude:: ../../../examples/cookbook/recipe_move_to_managed.cpp
                    :lines: 26

The move operation will copy the data from host memory to unified memory,
allocated using the provided ``um_allocator``. The original allocation in host
memory will be deallocated. The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_move_to_managed.cpp
