API v2 Design Rationale
=======================

Overview
--------

Umpire's API v2 work adds a modern C++ layer on top of the existing library
without replacing the established v1 interfaces. The intent is to preserve
tooling and backend compatibility while making it easier to express memory
resources, strategies, and typed allocation in ordinary C++ code.

This document summarizes the design rationale from
``openspec/changes/add-api-v2/design.md`` for users who want the motivations
behind the API shape without reading the full change proposal.

Core Goals
----------

API v2 is designed to provide:

- zero-cost abstractions where compile-time information can remove runtime work
- STL-compatible typed allocators for standard containers and algorithms
- explicit, composable memory resources and strategies instead of string-based
  factory configuration
- backward compatibility with existing Umpire tooling and operation backends
- opt-in thread safety and opt-out tracking overhead

Key Decisions
-------------

Registry-centered design
~~~~~~~~~~~~~~~~~~~~~~~~

API v2 uses ``umpire::detail::registry`` for allocator identity and allocation
tracking instead of routing every new interface through the v1
``ResourceManager`` factory workflow. This keeps global tracking available for
interop and tools while allowing new resources to be constructed directly.

Template-based memory hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The API combines a polymorphic ``memory`` base class with
``memory_resource<Platform>`` templates and concrete resource types such as
``host_memory``. This keeps the runtime model simple enough for shared tracking
while still propagating platform information at compile time.

Typed allocators
~~~~~~~~~~~~~~~~

``allocator<T, Memory>`` exists so Umpire-managed memory can be used directly
with ``std::vector``, ``std::map``, ``std::allocate_shared``, and other
allocator-aware library facilities. The design favors standard C++ allocator
semantics over a Umpire-specific container layer.

Platform tags and template dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Platform tag types such as ``host_platform`` and ``cuda_platform`` let the API
express intent in types. This enables compile-time dispatch and specialization
without forcing users to manually thread runtime platform enums through new
interfaces.

Composable strategies
~~~~~~~~~~~~~~~~~~~~~

API v2 strategies follow the decorator pattern. Wrappers such as
``thread_safe<Memory>``, ``fixed_pool<Memory>``, and ``size_limiter<Memory>``
can be stacked explicitly, making composition visible in user code instead of
being hidden inside factory strings or opaque configuration objects.

Shared allocation tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~

The design keeps v1 and v2 allocations visible through shared tracking
infrastructure so replay, introspection, and existing operations continue to
work across both APIs. That compatibility requirement heavily influenced the
registry and allocation record design.

Memory operations
~~~~~~~~~~~~~~~~~

API v2 adds template front-ends for operations such as ``copy``, ``memset``,
``prefetch``, and ``reallocate`` while keeping the underlying
``MemoryOperation`` implementations intact. The new layer improves type safety
and platform expression without rewriting backend operation code.

Trade-offs
----------

The API v2 design accepts a few costs in exchange for better ergonomics and
optimization opportunities:

- more templates means more compile-time surface area
- tracking-aware interoperability keeps some coupling to existing global state
- deeper strategy composition can produce more verbose types
- optional backends still rely on conditional compilation and target-specific
  validation

Even with those trade-offs, the design keeps the implementation aligned with
Umpire's HPC constraints: broad compiler support, optional backend enablement,
and compatibility with existing tooling.

Where To Look Next
------------------

- ``openspec/changes/add-api-v2/design.md`` for the full design record
- ``openspec/changes/add-api-v2/specs/api-v2/spec.md`` for the normative API v2
  requirements
- ``examples/api_v2/stl_containers.cpp`` for a small end-to-end example
- ``tests/integration/api_v2/`` for exercised API v2 behavior
