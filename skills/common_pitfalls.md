# Common Pitfalls

1. Accidentally synchronizing GPU.
2. Breaking host-only builds.
3. Introducing allocation tracking overhead in release builds.
4. Violating allocator equality semantics.
5. Using exceptions in device code.
6. Adding hidden memory ownership.
7. Breaking ABI with template signature changes.
