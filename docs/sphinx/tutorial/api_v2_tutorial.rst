API v2 Tutorial
================

Overview
--------

This tutorial walks through the current API v2 workflow from the simplest host
allocation to layered strategy composition. The host-side examples in this
document are backed by source files under ``examples/api_v2/`` and are safe to
build on this machine.

Device-specific examples for CUDA, HIP, SYCL, and OpenMP target are included
as code snippets only. This machine has no GPU or target-offload runtime, so
device validation remains a follow-up workflow rather than something claimed as
tested here.

Example 1: Basic Host Allocation
--------------------------------

Start with the default host singleton:

.. literalinclude:: ../../../examples/api_v2/basic_host_allocation.cpp
   :language: cpp
   :caption: examples/api_v2/basic_host_allocation.cpp

What this demonstrates:

- ``resource::host_memory<>::get()`` is the normal host entry point
- ``allocator<T, Memory>`` converts element counts into byte allocations
- STL containers can use the same allocator instance directly

Use this pattern when:

- you are starting new host-only API v2 code
- you want typed allocation instead of repeated ``sizeof(T)`` arithmetic
- you want a single memory source shared across raw allocation and containers

Pitfalls:

- the raw ``host_memory`` interface is byte-oriented, while
  ``allocator<T, Memory>`` is element-oriented
- ``allocator<T, Memory>::deallocate()`` takes an element count for STL
  compatibility, even though the underlying resource is pointer-based

Example 2: Copy Between Host Buffers
------------------------------------

API v2 operations use typed element counts for typed pointers:

.. literalinclude:: ../../../examples/api_v2/cross_platform_copy.cpp
   :language: cpp
   :caption: examples/api_v2/cross_platform_copy.cpp

What this demonstrates:

- ``umpire::memset(dst, value, count)`` operates on typed element counts
- ``umpire::copy(src, dst, count)`` uses the same typed convention
- the source file records both the host path and a guarded CUDA path
- standalone build validation for this example is tracked separately, while
  host operation semantics remain covered by the API v2 test suite

Use this pattern when:

- you want a small, typed replacement for raw byte copies
- you are writing helper code that should extend naturally to non-host
  backends later

Pitfalls:

- for typed pointers, ``count`` means elements, not bytes
- for ``void*`` operations, the size parameter is bytes

Example 3: Fixed-Size Pooling
-----------------------------

``fixed_pool`` is a good fit when allocation size is stable:

.. literalinclude:: ../../../examples/api_v2/fixed_pool.cpp
   :language: cpp
   :caption: examples/api_v2/fixed_pool.cpp

What this demonstrates:

- strategies wrap an existing resource instead of being created through
  ``ResourceManager``
- the pool size is expressed directly in ordinary C++ constructor arguments
- ``release()`` can return fully free backing pools to the parent resource

Use this pattern when:

- you allocate the same object size repeatedly
- allocator churn matters more than flexibility

Pitfalls:

- ``fixed_pool`` rejects any request whose size differs from the configured
  object size
- if sizes vary, prefer ``quick_pool`` or ``dynamic_pool_list`` instead

Example 4: Thread-Safe Shared Resource
--------------------------------------

Wrap a shared resource in ``thread_safe`` when several threads may call it at
the same time:

.. literalinclude:: ../../../examples/api_v2/thread_safe.cpp
   :language: cpp
   :caption: examples/api_v2/thread_safe.cpp

What this demonstrates:

- thread safety is explicit and opt-in
- the wrapper preserves the parent platform type
- the wrapped host allocation is still directly usable from the CPU

Use this pattern when:

- one resource instance is shared by multiple threads
- simplicity matters more than the cost of serialized allocation calls

Pitfalls:

- ``thread_safe`` protects only calls that go through the wrapper
- if each thread can own its own resource instance, that is often cheaper than
  sharing one locked wrapper

Example 5: Strategy Composition
-------------------------------

Strategies layer naturally from inner resource to outer wrapper:

.. code-block:: cpp

   #include "umpire/resource/host_memory.hpp"
   #include "umpire/strategy/fixed_pool.hpp"
   #include "umpire/strategy/thread_safe.hpp"

   using host_memory = umpire::resource::host_memory<>;
   using pool_type = umpire::strategy::fixed_pool<host_memory>;
   using shared_pool = umpire::strategy::thread_safe<pool_type>;

   int main()
   {
     auto& host = host_memory::get();
     pool_type pool{"PARTICLES", &host, sizeof(double), 256};
     shared_pool safe_pool{"SHARED_PARTICLES", &pool};

     double* value = static_cast<double*>(safe_pool.allocate(sizeof(double)));
     *value = 42.0;
     safe_pool.deallocate(value);
   }

What this demonstrates:

- composition is visible in the type
- the innermost object is the real backend resource
- outer wrappers add policy without changing the parent backend

Use this pattern when:

- you want to build a small allocation pipeline out of orthogonal behaviors
- you want the allocator configuration to be obvious in code review

Pitfalls:

- order matters: ``thread_safe<fixed_pool<host_memory>>`` is different from
  wrapping the layers in another order
- deeper composition can produce verbose types, so local ``using`` aliases help

Example 6: Device Resources
---------------------------

The same API shape extends to non-host resources, but these examples are
documented only on this machine.

CUDA:

.. code-block:: cpp

   #include "umpire/resource/cuda_device_memory.hpp"

   int main()
   {
     auto& gpu = umpire::resource::cuda_device_memory<>::get();
     float* values = static_cast<float*>(gpu.allocate(sizeof(float) * 256));
     gpu.deallocate(values);
   }

SYCL:

.. code-block:: cpp

   #include "umpire/resource/sycl_device_memory.hpp"
   #include "umpire/util/sycl_compat.hpp"

   int main()
   {
     sycl::queue queue;
     umpire::resource::sycl_device_memory<> device{"SYCL_DEVICE", queue};
     void* ptr = device.allocate(4096);
     device.deallocate(ptr);
   }

OpenMP target:

.. code-block:: cpp

   #include "umpire/resource/openmp_target_memory.hpp"

   int main()
   {
     auto& target = umpire::resource::openmp_target_memory<>::get();
     void* ptr = target.allocate(4096);
     target.deallocate(ptr);
   }

These snippets are useful as starting points, but hardware-backed validation is
tracked separately in Beads and should be completed on target-capable systems.

Where To Go Next
----------------

- See :doc:`../features/api_v2` for the broader user guide.
- See :doc:`../features/api_v2_migration` for incremental migration advice from
  v1.
- See ``examples/api_v2/stl_containers.cpp`` for additional container-focused
  usage.
