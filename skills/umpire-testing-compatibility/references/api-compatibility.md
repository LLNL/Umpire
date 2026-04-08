# API and ABI Compatibility

## Public Surface

- Treat installed headers under `include/umpire/**` as public API.
- In the source tree, that includes top-level headers in `src/umpire/*.hpp` and installed subdirectory headers such as `src/umpire/strategy/*.hpp`, `src/umpire/resource/*.hpp`, `src/umpire/op/*.hpp`, and `src/umpire/alloc/*.hpp`.
- If you are unsure whether a header is public, check the relevant `install(FILES ...)` entry in the CMake files before changing semantics.

## Breaking Changes

- Removing or renaming public methods
- Changing public method signatures
- Changing default behavior or allocator semantics
- Changing class layout or virtual function sets in ways that affect ABI
- Silently changing exception behavior or user-visible invariants

## Safer Changes

- Internal implementation changes that preserve public behavior
- Documentation improvements
- New overloads or new types that do not replace existing behavior
- Bug fixes that preserve the intended public contract

## Release Expectations

- Document user-visible compatibility-impacting changes in `RELEASE_NOTES.md`.
- Use deprecation markers and migration guidance when removing or replacing established behavior.
- Follow semantic versioning expectations for major, minor, and patch releases.

## Practical Rule

- If a downstream user could notice the change by recompiling against installed headers or by linking against the updated library, treat it as a compatibility review item.
