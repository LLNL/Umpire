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
- selected ownership-changing v1 operations on bridged v2 host allocations now
  have focused interoperability coverage
- non-host legacy delegation for CUDA, HIP, SYCL, and OpenMP target resources
  remains backend-specific work and is not implied by the host-safe results

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
- tracked v2 ``resource::host_memory<>::get()`` allocation/deallocation
  lifecycles can be recorded and replayed through the existing replay toolchain
- v1 ``ResourceManager::memset()`` works on v2 host allocations
- v1 ``ResourceManager::copy()`` works across v1/v2 host allocations

That means mixed migration is practical for ordinary host-side movement and
inspection paths.

These results do not extend automatically to CUDA, HIP, SYCL, or OpenMP target
API v2 resources. The current compatibility bridge in ``src/umpire/memory.cpp``
only mirrors the canonical ``HOST`` resource into the legacy v1 allocation map.
Tracked non-host API v2 allocations still participate in the shared v2
registry, but legacy v1 entry points that discover ownership, size, or platform
through ``ResourceManager::m_allocations`` remain separate backend-specific
work.

The following remains deferred:

- ownership-changing v1 operations on v2 allocations, such as broader
  reallocation/deallocation flows that rely on full shared ownership semantics
- legacy v1 replay event coverage for v2-backed allocations beyond direct
  tracked HOST allocation/deallocation lifecycle
- backend-specific non-host legacy entry points that still assume the v1
  allocation map for ownership or size lookup

The host-safe implementation work is tracked separately as ``umpire-8zd``.
Replay follow-up work is tracked separately as ``umpire-ifd.5``.
Backend-capable non-host follow-up work is tracked separately as
``umpire-rhg``.

Authoritative Compatibility Matrix
----------------------------------

The table below is the release-facing summary of the mixed v1/v2 support
boundary on this branch.

.. list-table::
   :header-rows: 1

   * - Mixed API scenario
     - Status
     - Evidence
     - Notes
   * - v1 lookup, inspection, and leak-reporting on tracked v2 HOST allocations
     - Supported
     - ``api_v2_v1_interop_tests``, ``umpire-7uq``
     - Covers allocation visibility, ``hasAllocator``, ``findAllocationRecord``,
       ``getAllocator(ptr)``, allocator record reporting, and leak views.
   * - v1 ``ResourceManager::copy()`` and ``memset()`` on tracked v2 HOST allocations
     - Supported
     - ``api_v2_v1_interop_tests``
     - Published for host-safe mixed operation paths only.
   * - v1 host-safe ownership-changing operations on tracked v2 HOST allocations
     - Supported
     - ``api_v2_v1_interop_tests``, ``umpire-8zd``
     - Includes ``deallocate()``, zero-size ``reallocate()``, host-preserving
       ``move()``, and host-preserving ``reallocate(..., HOST)`` including async
       variants.
   * - v1 distinct-allocator reallocate and offset-pointer ownership-changing
       operations on tracked v2 HOST allocations
     - Rejected by design
     - ``api_v2_v1_interop_tests``, ``umpire-8zd``
     - These cases fail explicitly instead of silently changing ownership.
   * - Replay of direct tracked v2 HOST allocation/deallocation lifecycle
     - Supported
     - ``replay_tests``, ``replay_api_v2_host_audit``, ``umpire-ifd.2``
     - Current replay guarantee is limited to allocation lifecycle recording and
       replay for tracked HOST resources.
   * - Replay of high-level legacy v1 ``copy()``, ``memset()``, ``reallocate()``,
       and ``move()`` events for pure v1 (non-v2-tracked) allocations
     - Supported
     - ``replay_tests``, ``umpire-ifd.5``
     - The default new-ops path (``UMPIRE_RM_USE_NEW_OPS=On``) now emits the
       same high-level "copy", "memset", "reallocate", and "move" operation
       events as the legacy dispatch path, restoring replay parity for
       ``ResourceManager`` entry points that do not involve v2-tracked
       pointers.
   * - Replay of high-level legacy v1 copy, move, and reallocate events on
       v2-backed HOST allocations
     - Intentionally unsupported
     - ``replay_api_v2_host_audit``, ``umpire-ifd.5``
     - Tracked v2 HOST allocations already emit allocate/deallocate lifecycle
       events from the shared v2 registry; a redundant high-level operation
       event is deliberately withheld to avoid double-emission during replay.
   * - API v2 operation templates on API v2 HOST allocations
     - Supported
     - ``api_v2_operations_tests``
     - Host-side ``copy``, ``memset``, ``prefetch``, and ``reallocate`` are
       covered for direct API v2 HOST usage.
   * - API v2 operation templates acting on pointers originating from legacy v1
       allocations
     - No separate support claim
     - none
     - This branch does not publish dedicated validation for v2 operation APIs
       acting on v1-owned pointers beyond the shared host backend behavior.
   * - Legacy v1-on-v2 behavior on CUDA, HIP, SYCL, and OpenMP target
       allocations
     - Backend-specific, not yet published
     - ``umpire-vc7``, ``umpire-rhg``
     - Host-safe conclusions do not extend automatically to tracked non-host API
       v2 allocations.

Delegation Matrix Summary
-------------------------

The detailed audit below remains the implementation-facing source of truth.
This summary captures the migration stance engineers and release notes should
cite.

.. list-table::
   :header-rows: 1

   * - Legacy API surface
     - Current stance
     - Why
   * - Factory, lookup-by-name/id/resource, and allocator identity APIs
     - Remain native v1
     - They are fundamentally about legacy allocator handles and registry
       identity, not v2 ownership.
   * - Pointer lookup and inspection APIs
     - Remain native v1 over shared tracking
     - The compatibility goal is visibility of tracked v2 HOST allocations
       through existing tooling, not replacement of the v1 façade.
   * - Host-side ``copy()`` and ``memset()``
     - Remain native v1 façade over shared operation dispatch
     - Mixed host correctness is what matters; forcing these calls through a
       synthetic v2 wrapper does not add value.
   * - Host-safe ownership-changing ``deallocate()``, zero-size
       ``reallocate()``, host-preserving ``move()``, and host-preserving
       ``reallocate(..., HOST)``
     - Delegate to API v2-backed ownership
     - Tracked v2 HOST allocations already have a clear owner in the shared v2
       registry and preserve semantics under these paths.
   * - Distinct-allocator and offset-pointer ownership-changing cases
     - Intentionally unsupported
     - The current branch rejects these explicitly rather than silently
       changing ownership models.
   * - Non-host legacy delegation
     - Separate backend-specific work
     - Shared tracking exists, but ownership lookup, operation dispatch, and
       replay semantics are not yet published across device/offload backends.

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

Backend-Specific Non-Host Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The host-safe classifications above must not be generalized to tracked API v2
CUDA, HIP, SYCL, or OpenMP target allocations. The current implementation
splits into four backend-sensitive categories:

- ``ResourceManager::deallocate(ptr)`` and the zero-size ``reallocate``
  overloads:
  in the ``UMPIRE_RM_USE_NEW_OPS`` path these first consult the shared API v2
  registry via ``find_v2_allocation`` and can therefore route tracked non-host
  allocations to their owning v2 resource or strategy. This is an
  implementation hook, not yet a published support claim, because no
  backend-capable interoperability coverage exercises those legacy entry points
  on non-host v2 allocations.
- owner-preserving ``umpire::reallocate`` and
  ``ResourceManager::reallocate(ptr, size)``:
  the v2 path performs allocate-copy-free on the existing owner and dispatches
  same-platform copy through the API v2 operation layer. The resulting
  semantics are backend-specific because correctness depends on the compiled
  device/offload copy specializations, runtime behavior, and async lifetime
  ordering on real hardware.
- allocator-selected ``ResourceManager::move`` and
  ``ResourceManager::reallocate(..., Allocator)``:
  these still resolve the source owner through ``getAllocator(ptr)`` or the v1
  ``m_allocations`` map. Because only the canonical ``HOST`` resource is
  mirrored into that map today, these overloads are not currently safe to
  classify as supported for non-host v2 allocations.
- legacy ``ResourceManager::copy``, ``memset``, and ``prefetch`` on non-host
  v2 allocations:
  even in the new-ops path, these entry points and the generic operation
  callers still derive size and platform information from the v1 allocation
  map. Until those paths consult the shared v2 registry directly or non-host
  v2 allocations are bridged into ``m_allocations``, host conclusions must not
  be extended to device or offload resources.

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
  ``umpire-ifd.2``; direct tracked HOST allocation/deallocation replay is now
  validated locally, while broader legacy v1-on-v2 replay semantics remain
  separate follow-up work in ``umpire-ifd.5``
- introspection and leak-reporting validation for v2-backed allocations is
  tracked separately in ``umpire-7uq``
- host-side safe delegation implementation is tracked separately in
  ``umpire-8zd``
- representative workload signoff is tracked separately in ``umpire-ifd.4``,
  ``umpire-v0s``, and ``umpire-ds5``
- backend-specific non-host delegation semantics remain separate work because
  they require backend-aware ownership and validation beyond this host-only
  migration audit; concrete follow-on work is tracked in ``umpire-rhg``

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
