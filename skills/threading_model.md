# Threading Model

Thread-safety guarantees:
- ResourceManager: Thread-safe (uses internal locking)
- Allocators: Must not introduce race conditions
- Strategies: Must document thread-safety guarantees

Rules:
- No static non-const globals (except ResourceManager)
- No global mutable state
- Document any locks or synchronization points
- Avoid adding new locks in hot paths

Common patterns:
- Multiple threads can safely call ResourceManager::getInstance()
- Multiple threads can allocate from same Allocator (if strategy supports it)
- Pool strategies may use internal locks (document this)

When adding thread-safety:
- Verify no data races with ThreadSanitizer
- Document thread-safety level in class documentation
- Consider lock-free alternatives for hot paths
