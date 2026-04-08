# API Compatibility

Umpire is a library used by many HPC applications. API stability is critical.

Public API rules:
- Do NOT break public API without explicit approval
- Public API = headers in src/umpire/*.hpp (not src/umpire/*/*)
- Core public APIs: Allocator, ResourceManager, TypedAllocator

Breaking changes include:
- Removing public methods
- Changing method signatures
- Changing default behavior
- Removing allocator types
- Changing exception behavior
- Modifying allocator semantics

ABI stability:
- Template signature changes break ABI
- Virtual function changes break ABI
- Class layout changes break ABI
- Do not add virtual functions to existing classes
- Do not reorder class members

Safe changes:
- Adding new classes
- Adding new methods (overloads, not replacements)
- Adding new allocator strategies
- Internal implementation changes (no public API change)
- Documentation improvements
- Bug fixes that don't change semantics

Deprecation process:
- Mark deprecated features with [[deprecated]]
- Document replacement in deprecation message
- Keep deprecated features for at least one major version
- Announce in RELEASE_NOTES.md

Versioning:
- Follow semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Breaking API changes
- MINOR: New features, backward compatible
- PATCH: Bug fixes, backward compatible

When uncertain about compatibility:
- Ask before making changes
- Check if change affects public API
- Consider if downstream users will break
- Document changes in RELEASE_NOTES.md
