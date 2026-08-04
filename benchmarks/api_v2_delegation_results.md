# API v2 Delegation Overhead Benchmark Results

Benchmark: `benchmarks/api_v2_delegation_benchmarks.cpp` (target
`api_v2_delegation_benchmarks`).

Measures allocate/deallocate round-trip latency for:
- (a) v1 `rm.getAllocator("HOST")` raw
- (b) v1 `ThreadSafeAllocator` via `makeAllocator` (native v1 impl, or
  delegated to a v2 `thread_safe<v1_backed_memory>` when
  `UMPIRE_V1_DELEGATE_TO_V2` is enabled)
- (c) v1 `SizeLimiter` via `makeAllocator` (native v1 impl, or delegated to
  a v2 `size_limiter<v1_backed_memory>` when `UMPIRE_V1_DELEGATE_TO_V2` is
  enabled)
- (d) v2 `host_memory<>` direct

across sizes {64B, 4KB, 1MB}. The same binary source builds and runs
identically in both flag configurations; only the underlying v1 strategy
implementation differs. Command used to collect the numbers below:

```
./bin/api_v2_delegation_benchmarks --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
```

## Machine context

- Machine: Apple M1 Max, macOS 26.5.2 (Darwin 25.5.0, arm64)
- Compiler: Homebrew LLVM 19.1.7 (`/opt/homebrew/opt/llvm@19/bin/clang++`), `-stdlib=libc++`
- Build type: `Release`
- Configure commands:
  - `build-off`: `cmake -S . -B build-off -G Ninja -DCMAKE_C_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm@19/bin/clang++ -DCMAKE_CXX_FLAGS='-stdlib=libc++' -DUMPIRE_ENABLE_TESTS=On -DUMPIRE_ENABLE_TOOLS=On -DUMPIRE_ENABLE_DEVELOPER_BENCHMARKS=On -DUMPIRE_ENABLE_FILE_RESOURCE=On -DUMPIRE_ENABLE_SHARED_MEMORY=On -DENABLE_BENCHMARKS=On -DUMPIRE_ENABLE_BENCHMARKS=On -DENABLE_GBENCHMARK=On -DCMAKE_BUILD_TYPE=Release`
  - `build-on`: same, plus `-DUMPIRE_V1_DELEGATE_TO_V2=On`
- google/benchmark could not read `hw.cpufrequency` via `sysctl` on this
  host (harmless; only affects reported CPU-frequency metadata, not the
  timing measurements themselves).

## Results: `UMPIRE_V1_DELEGATE_TO_V2 = OFF` (native v1 strategies)

| Benchmark | 64 B | 4 KB | 1 MB |
|---|---|---|---|
| (a) v1 HOST raw | 90.9 ns | 97.7 ns | 182 ns |
| (b) v1 ThreadSafeAllocator | 103 ns | 110 ns | 198 ns |
| (c) v1 SizeLimiter | 92.6 ns | 99.4 ns | 184 ns |
| (d) v2 host_memory direct | 63.3 ns | 72.4 ns | 165 ns |

## Results: `UMPIRE_V1_DELEGATE_TO_V2 = ON` (v1 strategies delegate to v2)

| Benchmark | 64 B | 4 KB | 1 MB |
|---|---|---|---|
| (a) v1 HOST raw | 90.1 ns | 100 ns | 183 ns |
| (b) v1 ThreadSafeAllocator | 161 ns | 171 ns | 257 ns |
| (c) v1 SizeLimiter | 148 ns | 158 ns | 254 ns |
| (d) v2 host_memory direct | 66.8 ns | 71.7 ns | 160 ns |

All times are the mean of 3 repetitions (`--benchmark_repetitions=3`).

## Delegation overhead (delegated vs. native), (b) and (c)

Overhead = `(delegated - native) / native * 100`.

| Strategy | 64 B | 4 KB | 1 MB |
|---|---|---|---|
| (b) ThreadSafeAllocator | +56.3% | +55.5% | +29.8% |
| (c) SizeLimiter | +59.8% | +59.0% | +38.0% |

## Observations

- `(a)` v1 HOST raw and `(d)` v2 host_memory direct are effectively
  unchanged between the two configurations (both bypass the
  `v1_backed_memory` bridge entirely), confirming the overhead is isolated
  to the delegated strategies as expected.
- The delegated `ThreadSafeAllocator` and `SizeLimiter` paths add roughly
  30-60 ns of fixed overhead per allocate/deallocate round trip versus
  their native v1 implementations. This overhead is largest in relative
  terms at small sizes (56-60% at 64 B) and shrinks proportionally at
  1 MB (~30-38%) since the fixed per-call cost is a smaller fraction of
  the (already larger) total latency there; in absolute terms the added
  cost is roughly constant (approximately 55-70 ns) across all three
  sizes, consistent with the delegated path's extra registry
  registration/lookup and an additional `thread_safe`/`size_limiter`
  wrapper layer on top of the v1 `AllocationStrategy` call (see
  `src/umpire/strategy/detail/v1_backed_memory.hpp`).
- `v2 host_memory direct` is consistently the fastest path in both
  configurations, as it has no v1 bookkeeping (`ResourceManager`
  registration, allocator lookup by name, etc.) to maintain.
