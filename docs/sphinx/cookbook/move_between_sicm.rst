.. _move_between_sicm:

=====================================
Move Allocations Between SICM Devices
=====================================

SICM is a library that allows for the allocation of memory onto
heterogenous memory types with a unified interface. Arenas are
created on a device, and individual allocations are obtained from
arenas. Because of this, individual allocations cannot be moved.
Rather, entire arenas must be moved all at once. This allows for
the movement of multiple variables that should be located on the
same device at the same time without having to worry about moving
variables that should not be moved because they happened to be
allocated on the same pages, which could happen if the variables
were allocated and moved individually.

A variable can be allocated on a SICM device using the
:class:`umpire::strategy::SICMStrategy`. It can therefore also be
moved between SICM devices using the
:func:`umpire::ResourceManager::move` operation.

In this recipe, we create an allocation on a SICM device and move it to
another SICM device.

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_move_between_sicm_devices.cpp
