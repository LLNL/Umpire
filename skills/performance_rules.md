# Performance Requirements

Hot paths:
- allocate()
- deallocate()

Must avoid:
- heap allocations
- locking (unless required by strategy)
- iostream usage
- dynamic dispatch if avoidable

Prefer:
- inline functions
- constexpr when possible
- static polymorphism
