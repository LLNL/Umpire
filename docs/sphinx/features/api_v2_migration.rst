========================
API v2 Migration Guide
========================

Overview
--------

Umpire API v2 adds a typed, template-based layer alongside the existing v1
``ResourceManager`` workflow. Migration is incremental: v1 remains available,
and applications can mix v1 and v2 usage while they move code over in small
steps.

This guide focuses on the current branch state:

- v1 ``umpire::ResourceManager`` and ``umpire::Allocator`` remain valid
- v2 introduces the lowercase typed allocator template,
  ``umpire::allocator<T, Memory>``
- host-side v1/v2 interoperability is validated for allocation visibility plus
  v1 ``copy()`` and ``memset()`` on v2 host allocations
- ownership-changing v1 operations on v2 allocations are tracked separately in
  Beads under ``umpire-xco``

When To Migrate
---------------

Migrate to API v2 when you want:

- type-safe allocation without repeated ``void*`` casts
- standard-library container integration through an STL-compatible allocator
- explicit composition of resources and strategies in C++ types
- compile-time platform information in generic code

Stay on v1 when you need:

- existing code to remain unchanged for now
- the factory-oriented ``ResourceManager`` workflow throughout a code path
- legacy Umpire integrations that already depend on v1 handles and naming

The recommended path is incremental adoption rather than a one-shot rewrite.

Naming Differences
------------------

The most important naming distinction on this branch is:

- ``umpire::Allocator``: legacy v1 allocator handle class
- ``umpire::allocator<T, Memory>``: API v2 typed allocator template

That difference matters when reading examples, includes, and compiler errors.

Core Mapping
------------

Getting host memory
~~~~~~~~~~~~~~~~~~~

V1 usually starts from ``ResourceManager``:

.. code-block:: cpp

   auto& rm = umpire::ResourceManager::getInstance();
   auto alloc = rm.getAllocator("HOST");

V2 starts from the concrete resource singleton:

.. code-block:: cpp

   auto& host = umpire::resource::host_memory<>::get();

Typed allocation
~~~~~~~~~~~~~~~~

V1 allocates bytes and returns ``void*``:

.. code-block:: cpp

   auto& rm = umpire::ResourceManager::getInstance();
   auto alloc = rm.getAllocator("HOST");

   auto* values = static_cast<double*>(alloc.allocate(100 * sizeof(double)));
   alloc.deallocate(values);

V2 allocates element counts and returns typed pointers:

.. code-block:: cpp

   auto& host = umpire::resource::host_memory<>::get();
   umpire::allocator<double, umpire::resource::host_memory<>> alloc{&host};

   double* values = alloc.allocate(100);
   alloc.deallocate(values, 100);

STL containers
~~~~~~~~~~~~~~

V1 typically uses ``umpire::TypedAllocator``:

.. code-block:: cpp

   auto& rm = umpire::ResourceManager::getInstance();
   auto alloc = rm.getAllocator("HOST");
   umpire::TypedAllocator<int> typed_alloc{alloc};
   std::vector<int, umpire::TypedAllocator<int>> values{typed_alloc};

V2 uses the typed allocator directly:

.. code-block:: cpp

   using host_memory = umpire::resource::host_memory<>;

   auto& host = host_memory::get();
   umpire::allocator<int, host_memory> alloc{&host};
   std::vector<int, umpire::allocator<int, host_memory>> values{alloc};

   values.push_back(3);
   values.push_back(5);

This is the same pattern used in
``examples/api_v2/stl_containers.cpp``.

Pools and strategy composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

V1 composes behavior through ``ResourceManager::makeAllocator(...)``:

.. code-block:: cpp

   auto& rm = umpire::ResourceManager::getInstance();
   auto pool = rm.makeAllocator<umpire::strategy::FixedPool>(
     "pool", rm.getAllocator("HOST"), 64, 1024);

V2 composes behavior directly in C++ types:

.. code-block:: cpp

   auto& host = umpire::resource::host_memory<>::get();

   umpire::strategy::fixed_pool<umpire::resource::host_memory<>>
     pool{"pool", &host, 64, 1024};

   umpire::allocator<char, umpire::strategy::fixed_pool<umpire::resource::host_memory<>>>
     alloc{&pool};

This pattern is more explicit: the wrapped resource and the strategy are both
visible in the type.

Common Migration Patterns
-------------------------

Pattern 1: Replace byte math at call sites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert code like this:

.. code-block:: cpp

   auto* ptr = static_cast<MyType*>(alloc.allocate(count * sizeof(MyType)));

to this:

.. code-block:: cpp

   umpire::allocator<MyType, Memory> alloc{&memory};
   auto* ptr = alloc.allocate(count);

This removes repeated ``sizeof(...)`` bookkeeping from call sites.

Pattern 2: Move new code to direct resources first
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lowest-risk migration is often:

1. Keep existing v1 allocators where they already work.
2. Use ``resource::host_memory<>::get()`` for new host-only code.
3. Introduce typed allocators only where containers or type safety matter.

That preserves working v1 flows while letting new components adopt v2 locally.

Pattern 3: Migrate containers before custom allocation plumbing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your code already uses ``std::vector``, ``std::map``, ``std::deque``, or
``std::allocate_shared``, switching those call sites to
``umpire::allocator<T, Memory>`` is usually simpler than first rewriting
resource ownership across an entire subsystem.

Interoperability
----------------

The current host-only compatibility coverage validates:

- v2 ``resource::host_memory<>::get()`` allocations appear in the v1
  ``ResourceManager``
- v1 ``ResourceManager::memset()`` works on v2 host allocations
- v1 ``ResourceManager::copy()`` works across v1/v2 host allocations

That means mixed migration is practical for ordinary host-side movement and
inspection paths.

The following remains deferred:

- ownership-changing v1 operations on v2 allocations, such as broader
  reallocation/deallocation flows that rely on full shared ownership semantics

That deferred work is tracked separately as ``umpire-xco``.

Recommended Migration Order
---------------------------

1. Start with host-only code paths.
2. Replace raw byte-count allocation sites with typed v2 allocators.
3. Migrate container-heavy code to ``umpire::allocator<T, Memory>``.
4. Introduce strategy composition explicitly where pools or wrappers are
   already conceptually present.
5. Keep backend-specific validation on hardware-capable systems as a separate
   step.

Deprecation Timeline
--------------------

The API v2 proposal describes a staged adoption plan rather than immediate v1
removal:

- months 0-4: implementation and internal testing
- months 4-6: beta release and feedback
- months 6-18: v1 and v2 supported in parallel, with v1 deprecated in
  documentation
- month 18+: v1 sunset planning

On this branch, that means v1 remains a supported migration partner rather than
something that must be removed before adopting v2.

Practical Advice
----------------

- Prefer lowercase ``umpire::allocator`` for new typed code.
- Prefer direct resource singletons such as
  ``umpire::resource::host_memory<>::get()`` for new host-side API v2 entry
  points.
- Keep existing ``ResourceManager``-based subsystems stable until there is a
  clear benefit to moving them.
- Validate GPU-backed migration separately on capable systems; this machine's
  host-only coverage does not substitute for device validation.

Related References
------------------

- ``docs/sphinx/features/api_v2_design_rationale.rst``
- ``examples/api_v2/stl_containers.cpp``
- ``tests/integration/api_v2/test_v1_v2_interop.cpp``
- ``openspec/changes/add-api-v2/design.md``
- ``openspec/changes/add-api-v2/specs/api-v2/spec.md``
