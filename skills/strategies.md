# Allocation Strategies

Strategies are:
- Composable (can wrap other strategies)
- Policy layers between Allocator and MemoryResource
- May have different performance characteristics

Common strategies:
- Pools: DynamicPoolList, DynamicSizePool, QuickPool, FixedPool, MixedPool
- Advisors: AllocationAdvisor (memory hints)
- Prefetchers: AllocationPrefetcher
- Limiters: SizeLimiter
- Alignment: AlignedAllocator
- NUMA: NumaPolicy

When adding strategies:
- Document thread-safety guarantees
- Document performance implications (O(1), O(log n), etc.)
- Ensure composability is maintained
- Add examples showing usage
