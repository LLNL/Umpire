.. _file_allocation:

==================================
Using File System Allocator (FILE)
==================================

Umpire supports the use of file based memory allocation. When ``ENABLE_FILE_RESOURCE`` 
is enabled, the environment variables ``UMPIRE_MEMORY_FILE_DIR``can be used to determine 
where memory can be allocated from:

.. list-table::
   :widths: 25 25 40
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``UMPIRE_MEMORY_FILE_DIR``
     - ./
     - Directory to create and allocate file based allocations

Requesting the allocation takes two steps: 1) getting a "FILE" allocator, 
2) requesting the amount of memory to allocate.

.. literalinclude:: ../../../examples/cookbook/recipe_filesystem_memory_allocation.cpp
                    :lines: 12-15

To deallocate:

.. literalinclude:: ../../../examples/cookbook/recipe_filesystem_memory_allocation.cpp
                    :lines: 17

The complete example is included below:

.. literalinclude:: ../../../examples/cookbook/recipe_filesystem_memory_allocation.cpp
