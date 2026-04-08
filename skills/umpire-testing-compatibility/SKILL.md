---
name: umpire-testing-compatibility
description: Testing, API, and ABI guidance for Umpire. Use when adding or modifying unit, integration, or application tests, reviewing public API or ABI impact, or checking compatibility and release-note expectations for a change.
---

# Umpire Testing Compatibility

Keep coverage focused, portable, and fast while protecting public behavior and installed-header compatibility. Use this skill when a change needs tests or when a change might affect user-visible APIs, ABI, or downstream expectations.

## Workflow

1. Read [references/testing.md](references/testing.md) for test placement, coverage expectations, and test design rules.
2. Read [references/api-compatibility.md](references/api-compatibility.md) when the change touches installed headers, class layout, method signatures, or user-visible semantics.
3. Pair with `umpire-platform-backends` for backend-enabled coverage and configuration details.

## Operating Rules

- Add or update tests when behavior changes.
- Keep tests deterministic, small, and fast enough for CI.
- Treat installed headers as public surface unless proven otherwise.
- Ask before making breaking API, ABI, or semantic changes.
