.. _get_largest_available_block_in_pool:

=========================================================
Determining the Largest Block of Available Memory in Pool
=========================================================

The :class:`umpire::strategy::QuickPool` provides a
:func:`umpire::strategy::QuickPool::getLargestAvailableBlock` that may be
used to determine the size of the largest block currently available for
allocation within the pool.
To call this
function, you must get the pointer to the
:class:`umpire::strategy::AllocationStrategy` from the
:class:`umpire::Allocator`:

.. literalinclude:: ../../../examples/cookbook/recipe_get_largest_available_block_in_pool.cpp
   :start-after: _sphinx_tag_tut_unwrap_start
   :end-before: _sphinx_tag_tut_unwrap_end
   :language: C++

Once you have the pointer to the appropriate strategy, you can call the
function:

.. literalinclude:: ../../../examples/cookbook/recipe_get_largest_available_block_in_pool.cpp
   :start-after: _sphinx_tag_tut_get_info_start
   :end-before: _sphinx_tag_tut_get_info_end
   :language: C++

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_get_largest_available_block_in_pool.cpp
