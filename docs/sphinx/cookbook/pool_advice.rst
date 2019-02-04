.. _pool_advice:

=============================
Apply Memory Advice to a Pool
=============================

When using unified memory on systems with CUDA GPUs, various types of memory
advice can be applied to modify how the CUDA runtime moves this memory around
between the CPU and GPU. One type of advice that can be applied is "preferred
location", and you can specificy where you want the preferred location of the
memory to be. This can be useful for ensuring that the memory is kept on the
GPU.

By creating a pool on top of an :class:`umpire::strategy::AllocationAdvisor`,
you can amortize the cost of applying memory advice:

.. literalinclude:: ../../../examples/cookbook/recipe_pool_advice.cpp
                    :lines: 34-44

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_pool_advice.cpp
