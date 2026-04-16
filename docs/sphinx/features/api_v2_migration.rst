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

Legacy V1 Surface Audit
-----------------------

The current branch now has enough implementation detail to classify the v1
entry points that matter most for migration. This is an implementation-facing
audit of the existing ``ResourceManager`` and ``Allocator`` surface, not a
promise that every v1 path should be rewritten to call a v2 entry point.

The classifications used here are:

- ``Delegate to API v2``:
  ownership or operation semantics are already routed through API v2-backed
  behavior for bridged host allocations, or should be expanded in that
  direction without changing user-visible semantics
- ``Remain native v1``:
  the API is fundamentally about v1 handles, factories, aliases, or legacy
  bookkeeping and should continue to be implemented in v1 while relying on
  shared tracking where needed
- ``Intentionally unsupported``:
  the combination is not a current migration target and should stay explicitly
  documented rather than silently implied

ResourceManager migration surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+
| v1 entry point                                            | Classification     | Notes                                                       |
+===========================================================+====================+=============================================================+
| ``getAllocator(name)``, ``getAllocator(id)``,             | Remain native v1   | Factory and handle lookup stay centered on the legacy       |
| ``getAllocator(resource)``                                |                    | allocator registry. These are about v1 object identity, not |
|                                                           |                    | v2 backend ownership.                                       |
+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+
| ``getAllocator(ptr)``                                     | Remain native v1   | Should keep using shared allocation tracking to resolve a   |
|                                                           |                    | v1 handle for mixed code paths. The important requirement   |
|                                                           |                    | is visibility of v2-backed allocations, not replacing the   |
|                                                           |                    | returned type.                                              |
+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+
| ``hasAllocator(ptr)``, ``findAllocationRecord(ptr)``,     | Remain native v1   | These are legacy inspection APIs. They should continue to   |
| ``getSize(ptr)``                                          |                    | work through the shared map and compatibility bridge.       |
+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+
| ``copy(dst, src, size)`` and async overload               | Remain native v1   | The v1 façade already dispatches through the shared         |
|                                                           |                    | operation layer. Mixed v1/v2 host operation support is the  |
|                                                           |                    | compatibility requirement, not replacement of the façade.   |
+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+
| ``memset(ptr, value, size)`` and async overload           | Remain native v1   | Same rationale as ``copy``. The critical migration outcome  |
|                                                           |                    | is correct operation on v2-backed allocations.              |
+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+
| ``deallocate(ptr)``                                       | Delegate to API v2 | In the new-ops path, bridged v2 allocations already route   |
|                                                           |                    | to the owning v2 strategy for deallocation.                 |
+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+
| ``reallocate(ptr, 0)`` and async zero-size overload       | Delegate to API v2 | Bridged host allocations already release through the v2     |
|                                                           |                    | owner when the request becomes a deallocation.              |
+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+
| ``reallocate(ptr, size, HOST)`` and async overload        | Delegate to API v2 | Host-preserving reallocation on bridged host allocations    |
|                                                           |                    | should continue to preserve v2 ownership semantics.         |
+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+
| ``move(ptr, HOST)``                                       | Delegate to API v2 | Host-preserving move already short-circuits and keeps the   |
|                                                           |                    | v2 owner in place for bridged host allocations.             |
+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+
| ``move(ptr, distinct host allocator)``                    | Delegate to API v2 | The move flow already supports ownership transfer from a    |
|                                                           |                    | bridged v2 host allocation into a distinct v1 allocator.    |
+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+
| ``reallocate(ptr, size, distinct allocator)``             | Intentionally      | This remains a semantic boundary today. For bridged host    |
|                                                           | unsupported        | allocations the distinct-allocator path is rejected instead |
|                                                           |                    | of silently changing ownership.                             |
+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+
| offset-pointer ownership-changing operations              | Intentionally      | Offset-pointer reallocate and move continue to error out.   |
|                                                           | unsupported        | The correct behavior is explicit rejection, not implicit    |
|                                                           |                    | ownership conversion.                                       |
+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+
| ``prefetch`` on non-host resources                        | Intentionally      | This is backend-specific and should be handled as a         |
|                                                           | unsupported here   | separate follow-on migration task rather than inferred from |
|                                                           |                    | host interoperability.                                      |
+-----------------------------------------------------------+--------------------+-------------------------------------------------------------+

Allocator migration surface
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-----------------------------------------------+--------------------+-----------------------------------------------------------+
| v1 entry point                                | Classification     | Notes                                                     |
+===============================================+====================+===========================================================+
| ``Allocator::allocate`` / ``allocate(name)``  | Remain native v1   | These are still v1 handle operations and keep v1 tracking |
| / ``allocate(resource)``                      |                    | and event semantics.                                      |
+-----------------------------------------------+--------------------+-----------------------------------------------------------+
| ``Allocator::deallocate``                     | Remain native v1   | Pointer ownership is tied to the creating v1 allocator    |
|                                               |                    | handle. Unknown-pointer deallocation should still go      |
|                                               |                    | through ``ResourceManager::deallocate`` when mixed        |
|                                               |                    | ownership is expected.                                    |
+-----------------------------------------------+--------------------+-----------------------------------------------------------+
| ``Allocator::getSize``                        | Remain native v1   | Continues to rely on the shared allocation map.           |
+-----------------------------------------------+--------------------+-----------------------------------------------------------+
| ``getHighWatermark`` / ``getCurrentSize`` /   | Remain native v1   | Introspection remains about the legacy allocator handle   |
| ``getActualSize`` / ``getAllocationCount``    |                    | and its underlying strategy.                              |
+-----------------------------------------------+--------------------+-----------------------------------------------------------+
| ``getName`` / ``getId`` / ``getStrategyName`` | Remain native v1   | These are allocator-handle identity APIs.                 |
+-----------------------------------------------+--------------------+-----------------------------------------------------------+
| ``getParent`` / ``getAllocationStrategy`` /   | Remain native v1   | These expose v1 strategy structure and should stay        |
| ``getPlatform``                               |                    | native.                                                   |
+-----------------------------------------------+--------------------+-----------------------------------------------------------+

Prioritized Follow-On Delegation Work
-------------------------------------

The highest-value follow-on implementation work for the host-safe migration
path is:

1. preserve and expand the existing host-side ownership delegation paths in
   ``ResourceManager`` where v2-backed allocations already have a clear owning
   v2 strategy
2. keep lookup, inspection, and factory APIs stable as native v1 façades over
   shared tracking instead of forcing them through a synthetic v2 wrapper
3. document and maintain the explicit rejection behavior for distinct-allocator
   and offset-pointer cases unless a future design proves a safe ownership
   model

Deferred / Separate Follow-Up Areas
-----------------------------------

- replay validation for v2-backed allocations is tracked separately in
  ``umpire-ifd.2``
- introspection and leak-reporting validation for v2-backed allocations is
  tracked separately in ``umpire-7uq``
- host-side safe delegation implementation is tracked separately in
  ``umpire-8zd``
- representative workload signoff is tracked separately in ``umpire-ifd.4``,
  ``umpire-v0s``, and ``umpire-ds5``
- backend-specific non-host delegation semantics remain separate work because
  they require backend-aware ownership and validation beyond this host-only
  migration audit

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
