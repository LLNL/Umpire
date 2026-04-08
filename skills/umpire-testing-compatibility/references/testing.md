# Testing

## Test Placement

- `tests/unit/`: unit coverage for individual classes and low-level behavior
- `tests/integration/`: end-to-end or multi-component behavior
- `tests/applications/`: application-level tests

## Coverage Expectations

- Add or update unit tests when local behavior changes.
- Add integration coverage for feature-level behavior when unit tests alone are not enough.
- Keep host-only builds and tests working.
- Add backend-enabled coverage when the feature is backend-specific and practical to validate.

## Test Design Rules

- Keep tests deterministic and fast.
- Avoid large allocations.
- Avoid hardcoded device IDs.
- Avoid timing-sensitive assertions and nondeterministic behavior.
- Clean up all allocations and resources.
- Avoid device synchronization unless the test is explicitly validating synchronization-related behavior.

## Practical Patterns

- Validate allocator identity, equality, and reported size semantics where relevant.
- Exercise strategy-specific behavior with focused assertions instead of broad smoke tests.
- Test error conditions and edge cases when behavior changes there.
- Use backend-specific coverage only when the backend behavior is the point of the test.

## Practical Rule

- If a test failure would be hard to diagnose from a unit test alone, add the smallest integration test that proves the behavior.
