.. _strategies:

==========
Strategies
==========

Strategies are used in Umpire to allow custom algorithms to be applied when
allocating memory. These strategies can do anything, from providing different
pooling methods to speed up allocations to applying different operations to
every alloctaion.

Strategies can be composed to combine their functionality, allowing flexible
and reusable implementations of different components.

.. doxygennamespace:: umpire::strategy

.. doxygenclass:: umpire::AllocationAdvisor

.. doxygenclass:: umpire::DynamicPool

.. doxygenclass:: umpire::FixedPool

.. doxygenclass:: umpire::MonotonicAllocationStrategy

.. doxygenclass:: umpire::SlotPool
