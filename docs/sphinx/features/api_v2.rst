API v2 User Guide
=================

Overview
--------

Umpire API v2 adds a typed, template-based layer alongside the existing v1
``ResourceManager`` and ``Allocator`` interfaces. The goal is not to replace
v1 immediately. The goal is to let new code express memory resources,
strategies, and typed allocation directly in C++ while keeping Umpire's
existing tracking, tools, and backend operations usable.

On this branch, the practical entry points are:

- ``umpire::memory`` for the common polymorphic interface
- ``umpire::memory_resource<Platform, Allocator, Tracking>`` for typed resource
  implementations
- concrete resources such as ``umpire::resource::host_memory<>``
- strategy decorators such as ``thread_safe<Memory>`` and ``fixed_pool<Memory>``
- ``umpire::allocator<T, Memory>`` for STL containers and other allocator-aware
  C++ facilities

API v2 is most useful when you want explicit composition, better type checking,
and a direct path from a memory concept in your code to the Umpire object that
implements it.

Why API v2?
-----------

API v1 is centered on ``ResourceManager``, string-named allocators, and
byte-oriented allocation. That model remains useful, but it does not make
compile-time platform information or strategy composition especially visible in
user code.

API v2 changes that by making the key decisions part of the type system:

- resource kind is carried in the memory type
- platform information is available through a ``platform`` type alias
- strategy composition is explicit in nested template types
- typed allocation removes repeated ``sizeof(T)`` arithmetic at call sites
- tracking remains optional, so hot paths can opt out when appropriate

The migration model is incremental, not disruptive. Host-side v1/v2
interoperability is already validated on this branch for allocation
visibility, inspection/reporting, mixed host operations, and selected
ownership-changing v1 calls on bridged host allocations. Direct tracked HOST
allocation lifecycle replay is also validated. Broader legacy replay parity and
non-host mixed API claims remain separate follow-up work.

Transition Status
-----------------

The current branch supports an incremental migration model rather than a
flag-day API replacement:

- New host-only code can adopt ``resource::host_memory<>`` and
  ``umpire::allocator<T, Memory>`` directly.
- Existing v1 host code can interoperate with tracked v2 HOST allocations
  through the validated ``ResourceManager`` inspection, operation, and
  selected ownership-changing paths documented in the migration guide.
- Replay support is currently published for direct tracked HOST allocation
  lifecycle, not yet for every legacy v1 operation path acting on a v2-backed
  allocation.
- CUDA, HIP, SYCL, and OpenMP target mixed v1-on-v2 behavior remain
  backend-specific and should not be inferred from host-safe results.

Use :doc:`api_v2_migration` as the authoritative source for the current
compatibility matrix, delegation matrix, and follow-up gaps.

Core Concepts
-------------

Memory hierarchy
~~~~~~~~~~~~~~~~

API v2 uses a small hierarchy with both runtime and compile-time roles:

.. code-block:: text

   memory
     |
     +-- memory_resource<Platform, Allocator, Tracking>
     |      |
     |      +-- resource::host_memory<>
     |      +-- resource::cuda_device_memory<>      (when enabled)
     |      +-- resource::hip_device_memory<>       (when enabled)
     |      +-- resource::sycl_device_memory<>      (when enabled)
     |      +-- resource::openmp_target_memory<>    (when enabled)
     |
     +-- strategy::allocation_strategy
            |
            +-- strategy::thread_safe<Memory>
            +-- strategy::fixed_pool<Memory>
            +-- strategy::coalescing_pool_list<Memory>
            +-- strategy::binned_pool<Memory>
            +-- strategy::size_limiter<Memory>
            +-- strategy::monotonic_buffer<Memory>
            +-- strategy::named<Memory>

``memory`` provides the common interface for allocation, deallocation,
statistics, and platform reporting. ``memory_resource`` adds compile-time
platform information and allocator selection. Concrete resources implement
backend behavior. Strategies wrap another ``memory`` object and modify behavior
without changing the underlying backend.

Platform tags and dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~

Each typed resource propagates a ``platform`` alias, for example
``umpire::host_platform``. That makes generic code easier to specialize:

.. code-block:: cpp

   #include "umpire/platform.hpp"
   #include "umpire/resource/host_memory.hpp"

   #include <type_traits>

   template <typename Memory>
   constexpr bool is_host_memory()
   {
     return std::is_same_v<typename Memory::platform, umpire::host_platform>;
   }

Platform tags are also what API v2 operations use internally when dispatching
between host, CUDA, HIP, SYCL, and OpenMP target backends.

Tracking
~~~~~~~~

Tracking is enabled by default for the standard resource aliases. When tracking
is enabled, allocations are visible through the shared API v2 registry and can
participate in interoperability with existing tooling.

When tracking cost matters more than introspection, use a resource alias with
tracking disabled:

.. code-block:: cpp

   #include "umpire/resource/host_memory.hpp"

   umpire::resource::fast_host_memory fast_host{"FAST_HOST"};
   void* ptr = fast_host.allocate(4096);
   fast_host.deallocate(ptr);

Tracking is especially relevant for ``size_limiter`` and for mixed v1/v2 host
workflows, because those paths depend on allocation metadata being available.

Strategy composition
~~~~~~~~~~~~~~~~~~~~

Strategies are ordinary wrappers. You compose them by constructing the
innermost resource first and then wrapping outward:

.. code-block:: cpp

   #include "umpire/resource/host_memory.hpp"
   #include "umpire/strategy/fixed_pool.hpp"
   #include "umpire/strategy/thread_safe.hpp"

   using host_memory = umpire::resource::host_memory<>;
   using fixed_pool = umpire::strategy::fixed_pool<host_memory>;
   using safe_pool = umpire::strategy::thread_safe<fixed_pool>;

   host_memory& host = host_memory::get();
   fixed_pool pool{"HOST_POOL", &host, sizeof(double), 256};
   safe_pool shared_pool{"THREAD_SAFE_HOST_POOL", &pool};

The outermost type is the one clients see. The wrapped parent remains
accessible through ``get_parent()`` when a strategy needs to be inspected.

Basic Usage
-----------

Host memory
~~~~~~~~~~~

``host_memory`` is the simplest place to start:

.. code-block:: cpp

   #include "umpire/resource/host_memory.hpp"

   int main()
   {
     auto& host = umpire::resource::host_memory<>::get();

     void* bytes = host.allocate(1024);
     host.deallocate(bytes);
   }

This is byte-oriented and close to API v1 semantics, but it still participates
in the API v2 registry and type system.

Typed allocation
~~~~~~~~~~~~~~~~

Use ``umpire::allocator<T, Memory>`` when you want element counts instead of
raw byte counts:

.. code-block:: cpp

   #include "umpire/allocator.hpp"
   #include "umpire/resource/host_memory.hpp"

   using host_memory = umpire::resource::host_memory<>;

   int main()
   {
     auto& host = host_memory::get();
     umpire::allocator<double, host_memory> alloc{&host};

     double* values = alloc.allocate(128);
     values[0] = 3.14;
     alloc.deallocate(values, 128);
   }

The second argument to ``deallocate()`` is present for STL allocator
compatibility. API v2 deallocation itself is pointer-based.

STL containers
~~~~~~~~~~~~~~

API v2 allocators are designed to work directly with standard containers:

.. code-block:: cpp

   #include "umpire/allocator.hpp"
   #include "umpire/resource/host_memory.hpp"

   #include <map>
   #include <memory>
   #include <string>
   #include <vector>

   using host_memory = umpire::resource::host_memory<>;

   int main()
   {
     auto& host = host_memory::get();

     umpire::allocator<int, host_memory> int_alloc{&host};
     std::vector<int, umpire::allocator<int, host_memory>> values{int_alloc};
     values.push_back(3);
     values.push_back(5);

     using pair_type = std::pair<const int, std::string>;
     std::map<int, std::string, std::less<int>,
              umpire::allocator<pair_type, host_memory>> labels{
       umpire::allocator<pair_type, host_memory>{&host}};
     labels.emplace(3, "three");

     auto shared = std::allocate_shared<std::string>(
       umpire::allocator<std::string, host_memory>{&host},
       "umpire api v2");

     return static_cast<int>(values.size() + labels.size() + shared->size());
   }

The repository example ``examples/api_v2/stl_containers.cpp`` uses this exact
pattern.

Advanced Usage
--------------

Fixed-size pooling
~~~~~~~~~~~~~~~~~~

Use ``fixed_pool`` when allocation size is stable and frequent:

.. code-block:: cpp

   #include "umpire/resource/host_memory.hpp"
   #include "umpire/strategy/fixed_pool.hpp"

   using host_memory = umpire::resource::host_memory<>;
   using pool_type = umpire::strategy::fixed_pool<host_memory>;

   int main()
   {
     auto& host = host_memory::get();
     pool_type pool{"PARTICLE_POOL", &host, sizeof(double), 512};

     void* a = pool.allocate(sizeof(double));
     void* b = pool.allocate(sizeof(double));

     pool.deallocate(a);
     pool.deallocate(b);
     pool.release();
   }

``fixed_pool`` throws if the requested size differs from the configured object
size. That is intentional; use ``binned_pool`` or ``coalescing_pool_list`` when the
request size varies.

Thread safety
~~~~~~~~~~~~~

Resources and strategies do not become thread-safe automatically. Wrap a shared
memory source in ``thread_safe`` when multiple threads will call
``allocate()`` and ``deallocate()`` concurrently:

.. code-block:: cpp

   #include "umpire/resource/host_memory.hpp"
   #include "umpire/strategy/thread_safe.hpp"

   using host_memory = umpire::resource::host_memory<>;
   using safe_host = umpire::strategy::thread_safe<host_memory>;

   int main()
   {
     auto& host = host_memory::get();
     safe_host shared{"SHARED_HOST", &host};

     void* ptr = shared.allocate(256);
     shared.deallocate(ptr);
   }

When thread safety is not required, prefer unwrapped resources or per-thread
instances to avoid serialization.

Usage limits
~~~~~~~~~~~~

``size_limiter`` enforces a live-byte cap on top of another resource:

.. code-block:: cpp

   #include "umpire/resource/host_memory.hpp"
   #include "umpire/strategy/size_limiter.hpp"

   using host_memory = umpire::resource::host_memory<>;
   using limited_host = umpire::strategy::size_limiter<host_memory>;

   int main()
   {
     auto& host = host_memory::get();
     limited_host limited{"LIMITED_HOST", &host, 1024};

     void* ptr = limited.allocate(512);
     limited.deallocate(ptr);
   }

Because ``size_limiter`` uses allocation metadata on deallocation, tracked
parents are the practical default.

Operations
~~~~~~~~~~

API v2 also exposes typed operation front-ends. For host-host movement,
``umpire::copy`` and ``umpire::memset`` can be used directly:

.. code-block:: cpp

   #include "umpire/op.hpp"
   #include "umpire/resource/host_memory.hpp"

   int main()
   {
     auto& host = umpire::resource::host_memory<>::get();

     int* src = static_cast<int*>(host.allocate(sizeof(int) * 4));
     int* dst = static_cast<int*>(host.allocate(sizeof(int) * 4));

     umpire::memset(src, 0, 4);
     umpire::copy(src, dst, 4);

     host.deallocate(dst);
     host.deallocate(src);
   }

The same interface extends to device backends when those backends are enabled
and validated on capable systems.

Backend Notes
-------------

Host support is the best-tested API v2 path on this machine. The codebase also
contains typed resources for CUDA, HIP, SYCL, and OpenMP target builds:

- ``resource::cuda_device_memory<>``
- ``resource::hip_device_memory<>``
- ``resource::sycl_device_memory<>``
- ``resource::openmp_target_memory<>``

Those resource types are part of the public API and can be documented and built
conditionally, but this machine has no GPU or target-offload hardware. Device
validation is therefore tracked as follow-up Beads work rather than claimed as
complete host-side coverage.

Performance Guidance
--------------------

API v2 aims to keep abstractions explicit and inexpensive, but performance
still depends on picking the right composition:

- use ``host_memory`` directly when you want the simplest tracked host path
- use ``fast_host_memory`` when you explicitly want to disable tracking overhead
- use ``fixed_pool`` for stable, repeated allocation sizes
- use ``binned_pool`` for a small range of short-lived allocation sizes
- use ``coalescing_pool_list`` for broader variable-size workloads
- use ``thread_safe`` only when the wrapped resource is truly shared across
  threads

Benchmarking and zero-overhead validation are tracked separately in
``umpire-2m2`` because this machine can only exercise the host slice.

Best Practices
--------------

- Prefer direct resource singletons such as ``host_memory<>::get()`` for new
  host-side code.
- Prefer ``umpire::allocator<T, Memory>`` for container integration instead of
  open-coded ``sizeof(T)`` arithmetic.
- Build strategy composition from the innermost resource outward.
- Keep device-specific validation separate from host-only documentation and
  examples.
- Use the migration guide when moving existing v1 code incrementally.

Related Documentation
---------------------

- :doc:`api_v2_design_rationale`
- :doc:`api_v2_migration`
- :doc:`../tutorial/api_v2_tutorial`
- ``examples/api_v2/stl_containers.cpp``
- ``tests/integration/api_v2/test_stl_compatibility.cpp``
