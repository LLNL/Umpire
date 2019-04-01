.. _pinned_pool:

========================================
Building a Pinned Memory Pool in FORTRAN
========================================

In this recipe, we show you how to build a pool in pinned memory using Umpire's
FORTRAN API. These kinds of pools can be useful for allocating buffers to be
used in communication routines in various scientific applications.

Building the pool take two steps: 1) getting a base "PINNED" allocator, and 2)
creating the pool:

.. literalinclude:: ../../../examples/cookbook/recipe_pinned_pool.F
                    :lines: 27-32

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_pinned_pool.F
