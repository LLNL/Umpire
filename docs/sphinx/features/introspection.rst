.. _introspection:

=============
Introspection
=============

Umpire provides introspection capabilities that allow you to query information about
allocations at runtime. This includes finding which allocator was used to allocate a
pointer, retrieving the size of an allocation, and accessing other metadata associated
with memory allocations.

Umpire supports two different introspection mechanisms: **map-based introspection**
(the traditional approach) and **header-based introspection** (a newer alternative).
The choice between these mechanisms is made at compile time via CMake configuration.

---------------------------
Map-Based Introspection
---------------------------

Map-based introspection is the traditional approach that has been part of Umpire since
its initial release. It maintains a centralized map of all allocations in the
``ResourceManager``.

How It Works
------------

- All allocations are registered in a global ``AllocationMap`` (``m_allocations`` in ``ResourceManager``)
- The map is implemented using Judy arrays, a high-performance sparse array data structure
- When you query a pointer, Umpire searches the map to find the corresponding allocation record
- The map stores ``AllocationRecord`` objects containing metadata (size, allocator, strategy, etc.)
- Supports **interior pointer lookups**: you can query any address within an allocated region

Technical Details
-----------------

**Memory Overhead:**
  - Per-allocation: One ``AllocationRecord`` entry (~88 bytes) in the Judy array
  - Global: Judy array data structure overhead
  - Additional ``RecordList`` blocks when multiple allocations share the same base address

**Performance:**
  - Allocation/deallocation: O(log n) insertion/removal in Judy array
  - Introspection queries: O(log n) search time
  - Uses ``findOrBefore()`` to support interior pointer queries
  - Mutex synchronization required for thread safety

**Thread Safety:**
  - Fully thread-safe via global mutex (``std::mutex m_mutex``)
  - All map operations (insert, find, remove) acquire the lock
  - May experience contention in multi-threaded workloads

**Memory Type Support:**
  - All memory types supported (HOST, DEVICE, PINNED, UNIFIED, etc.)
  - Works uniformly across all allocators and strategies

-----------------------------
Header-Based Introspection
-----------------------------

Header-based introspection is a newer alternative that stores allocation metadata
directly in a header preceding each allocation. This approach can provide faster
introspection queries but has some important limitations.

How It Works
------------

- A 64-byte aligned header is placed immediately before each allocation
- The header contains a pointer to the allocation's metadata (``AllocationRecord``)
- Metadata is stored in a separate ``FixedMallocPool`` (not in the user's allocation)
- The user receives a pointer to memory starting 64 bytes after the base allocation
- Introspection queries directly read the header (O(1) operation, no search required)

Technical Details
-----------------

**Memory Overhead:**
  - Per-allocation: 64-byte header + ~88-byte metadata in pool = ~152 bytes total
  - Header is 64-byte aligned (optimized for CPU cache lines and GPU transfers)
  - Metadata pool overhead grows with number of allocations

**Performance:**
  - Allocation/deallocation: O(1) header read/write + pool operations
  - Introspection queries: O(1) - direct header access, no search
  - Significantly faster introspection than map-based approach
  - Reduced mutex contention (no global allocation map lock)

**Thread Safety:**
  - Metadata pool operations protected by mutex
  - Header reads are not synchronized (assumes single-threaded access per allocation)
  - Less contention than map-based for introspection queries

**Memory Type Support:**
  - **LIMITED** - Only host-accessible memory types:

    - Host memory (always supported)
    - CUDA Unified/Managed memory
    - HIP Managed memory
    - SYCL USM Shared memory

  - **NOT supported:** Pure device memory, OpenMP target memory
  - The function ``supportsHeaderIntrospection(strategy)`` determines compatibility

------------------------
Feature Comparison
------------------------

.. list-table:: Map-Based vs. Header-Based Introspection
   :widths: 30 35 35
   :header-rows: 1

   * - Feature
     - Map-Based
     - Header-Based
   * - **Interior Pointer Support**
     - ✓ Yes - can query any address within allocation
     - ✗ No - requires exact base pointer
   * - **Use-After-Free Detection**
     - ✓ Partial - map entry can persist after deallocation
     - ✗ No - reads freed memory (undefined behavior)
   * - **Memory Types**
     - ✓ All types (HOST, DEVICE, PINNED, etc.)
     - ⚠ Limited - host-accessible only
   * - **Introspection Performance**
     - O(log n) search in Judy array
     - O(1) direct header read
   * - **Allocation Performance**
     - O(log n) map insertion + mutex
     - O(1) header write + pool allocation
   * - **Memory Overhead (small alloc)**
     - ~88 bytes per allocation
     - ~152 bytes per allocation (64B header + 88B metadata)
   * - **Memory Overhead (large alloc)**
     - ~88 bytes per allocation
     - ~152 bytes per allocation (fixed)
   * - **Thread Safety**
     - Fully synchronized (global mutex)
     - Synchronized metadata pool, unsynchronized headers
   * - **Mutex Contention**
     - Higher (every operation locks global map)
     - Lower (metadata pool lock only)
   * - **Zero-Byte Allocations**
     - Returns unique non-null pointer
     - Returns ``nullptr``
   * - **Reallocate Operations**
     - ✓ Fully supported
     - ⚠ Limited support
   * - **Allocator Enumeration**
     - ✓ Can enumerate all allocations via map
     - ⚠ Limited - only untracked allocations enumerable
   * - **Runtime Configuration**
     - No (compile-time only)
     - No (compile-time only)

-------------------
Build Configuration
-------------------

The introspection mechanism is selected at compile time via a CMake option:

**To enable header-based introspection (default):**

.. code-block:: bash

   cmake ... -DUMPIRE_ENABLE_HEADER_INTROSPECTION=On ...

**To use map-based introspection:**

.. code-block:: bash

   cmake ... -DUMPIRE_ENABLE_HEADER_INTROSPECTION=Off ...

.. note::
   The introspection mechanism cannot be changed at runtime. You must rebuild
   Umpire with the desired CMake option to switch between methods.

--------------------------------------
Choosing an Introspection Method
--------------------------------------

**Use Header-Based Introspection When:**

- You primarily work with host-accessible memory (CPU allocations, unified memory)
- Introspection queries are frequent and performance-critical
- You always query allocations using the exact pointer returned by ``allocate()``
- Memory overhead of 64 bytes per allocation is acceptable
- You can tolerate limited functionality (no interior pointers, limited reallocate)

**Use Map-Based Introspection When:**

- You need to support all memory types (including pure device memory)
- You require interior pointer lookups (querying addresses within allocations)
- You need better use-after-free detection for debugging
- You want full ``reallocate()`` functionality
- You need to enumerate all active allocations
- Your application performs relatively few introspection queries

**General Recommendation:**

For most users, **map-based introspection** (``UMPIRE_ENABLE_HEADER_INTROSPECTION=Off``)
is recommended as the safer, more feature-complete option. Header-based introspection
should be considered an optimization for specific workloads where introspection
performance is critical and the limitations are acceptable.

--------------------------------------
Limitations and Considerations
--------------------------------------

Header-Based Introspection Limitations
---------------------------------------

.. warning::
   **No Interior Pointer Support**: Header-based introspection cannot lookup allocations
   from interior pointers. Operations like ``Copy::Offset`` and ``Memset::Offset`` are
   not supported. You must use the exact pointer returned by ``allocate()``.

.. warning::
   **Use-After-Free Vulnerability**: Calling ``getSize()`` or other introspection
   functions on a deallocated pointer reads freed memory, resulting in undefined behavior.
   The map-based approach properly throws an exception in this case.

.. warning::
   **Memory Type Restrictions**: Header-based introspection only works with host-accessible
   memory. Pure device allocations will fall back to untracked mode or may not work correctly
   with some strategies.

.. warning::
   **Increased Memory Overhead**: Every allocation adds 64 bytes overhead. For applications
   with many small allocations (e.g., thousands of 64-byte allocations), this represents
   100% memory overhead.

.. warning::
   **Limited Reallocate Support**: Some reallocate operations are not fully supported and
   may exhibit undefined behavior. Thorough testing is recommended if using reallocate
   with header introspection.

.. warning::
   **No Alignment Validation**: Large alignment requests (>64 bytes) with ``AlignedAllocator``
   may not be correctly aligned. This can cause crashes on architectures with strict
   alignment requirements.

Map-Based Introspection Limitations
------------------------------------

.. note::
   **Performance Overhead**: Every allocation, deallocation, and introspection query
   requires a Judy array operation with O(log n) complexity and mutex synchronization.
   This may impact performance in high-frequency allocation workloads.

.. note::
   **Mutex Contention**: The global allocation map is protected by a mutex. Multi-threaded
   applications may experience contention when multiple threads allocate or query memory
   simultaneously.

Common Considerations
---------------------

.. note::
   **Untracked Allocators**: Allocators created with ``Tracking::Untracked`` bypass all
   introspection mechanisms (both map and header-based). This is useful for performance-critical
   paths where introspection is not needed.

.. note::
   **Compile-Time Decision**: The introspection method is selected at compile time and
   cannot be changed at runtime. All allocators in your application will use the same
   introspection mechanism.

Usage Examples
--------------

Basic introspection is identical regardless of which mechanism is used:

.. code-block:: c++

   #include "umpire/ResourceManager.hpp"
   #include "umpire/Allocator.hpp"

   auto& rm = umpire::ResourceManager::getInstance();
   auto allocator = rm.getAllocator("HOST");

   // Allocate memory
   void* ptr = allocator.allocate(1024);

   // Query allocation information
   std::size_t size = rm.getSize(ptr);  // Returns 1024
   umpire::Allocator alloc = rm.getAllocator(ptr);  // Returns the allocator used

   // With map-based introspection, interior pointers work:
   #ifndef UMPIRE_ENABLE_HEADER_INTROSPECTION
   void* interior_ptr = static_cast<char*>(ptr) + 512;
   std::size_t size2 = rm.getSize(interior_ptr);  // Also returns 1024
   #endif

   // Deallocate
   allocator.deallocate(ptr);

   // Map-based: This throws an exception (pointer not found)
   // Header-based: Undefined behavior (reads freed memory)
   // NEVER DO THIS - shown for illustration only:
   // std::size_t bad = rm.getSize(ptr);  // UNSAFE!

Checking Memory Type Support
-----------------------------

You can check if a strategy supports header introspection:

.. code-block:: c++

   #include "umpire/util/allocation_metadata.hpp"

   auto& rm = umpire::ResourceManager::getInstance();
   auto allocator = rm.getAllocator("DEVICE");

   #ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
   bool supports_header = umpire::util::supportsHeaderIntrospection(
       allocator.getAllocationStrategy()
   );

   if (!supports_header) {
       // This allocator will use untracked mode or may not work correctly
       std::cout << "Warning: Allocator does not support header introspection" << std::endl;
   }
   #endif
