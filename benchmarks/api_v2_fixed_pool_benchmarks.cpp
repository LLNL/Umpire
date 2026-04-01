//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/fixed_pool.hpp"
#include "umpire/resource/host_memory.hpp"

#include <benchmark/benchmark.h>
#include <cstdlib>
#include <vector>

namespace {

// Simple memory wrapper for direct malloc/free comparison
class malloc_memory : public umpire::memory {
public:
  malloc_memory() : umpire::memory{"malloc"} { }

  void* allocate(std::size_t size) override {
    return std::malloc(size);
  }

  void deallocate(void* ptr) override {
    std::free(ptr);
  }

  umpire::resource::Platform get_platform() const override {
    return umpire::resource::Platform::host;
  }
};

} // namespace

// ============================================================================
// Allocation Speed Benchmarks
// ============================================================================

// Benchmark: fixed_pool allocation speed
static void BM_fixed_pool_allocate(benchmark::State& state) {
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::fixed_pool<umpire::resource::host_memory<>>
    pool("bench_pool", &host, state.range(0));

  for (auto _ : state) {
    void* ptr = pool.allocate(state.range(0));
    benchmark::DoNotOptimize(ptr);
    pool.deallocate(ptr);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_fixed_pool_allocate)->Range(8, 8192)->Unit(benchmark::kNanosecond);

// Benchmark: direct malloc allocation speed (for comparison)
static void BM_malloc_allocate(benchmark::State& state) {
  for (auto _ : state) {
    void* ptr = std::malloc(state.range(0));
    benchmark::DoNotOptimize(ptr);
    std::free(ptr);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_malloc_allocate)->Range(8, 8192)->Unit(benchmark::kNanosecond);

// ============================================================================
// Deallocation Speed Benchmarks
// ============================================================================

// Benchmark: fixed_pool deallocation speed
static void BM_fixed_pool_deallocate(benchmark::State& state) {
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::fixed_pool<umpire::resource::host_memory<>>
    pool("bench_pool", &host, state.range(0), 10000);

  // Pre-allocate objects
  std::vector<void*> ptrs;
  ptrs.reserve(state.max_iterations);
  for (int64_t i = 0; i < state.max_iterations; ++i) {
    ptrs.push_back(pool.allocate(state.range(0)));
  }

  size_t idx = 0;
  for (auto _ : state) {
    pool.deallocate(ptrs[idx++]);
  }

  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_fixed_pool_deallocate)->Range(8, 8192)->Unit(benchmark::kNanosecond);

// Benchmark: direct free deallocation speed (for comparison)
static void BM_free_deallocate(benchmark::State& state) {
  // Pre-allocate objects
  std::vector<void*> ptrs;
  ptrs.reserve(state.max_iterations);
  for (int64_t i = 0; i < state.max_iterations; ++i) {
    ptrs.push_back(std::malloc(state.range(0)));
  }

  size_t idx = 0;
  for (auto _ : state) {
    std::free(ptrs[idx++]);
  }

  state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_free_deallocate)->Range(8, 8192)->Unit(benchmark::kNanosecond);

// ============================================================================
// Churn Benchmarks (Allocation/Deallocation Patterns)
// ============================================================================

// Benchmark: fixed_pool allocation/deallocation churn
static void BM_fixed_pool_churn(benchmark::State& state) {
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::fixed_pool<umpire::resource::host_memory<>>
    pool("bench_pool", &host, 64);

  for (auto _ : state) {
    std::vector<void*> ptrs;
    ptrs.reserve(state.range(0));

    // Allocate
    for (int64_t i = 0; i < state.range(0); ++i) {
      ptrs.push_back(pool.allocate(64));
    }

    // Deallocate
    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }

  state.SetItemsProcessed(state.iterations() * state.range(0) * 2);  // 2x for alloc+dealloc
}
BENCHMARK(BM_fixed_pool_churn)->Range(10, 1000)->Unit(benchmark::kMicrosecond);

// Benchmark: malloc/free churn (for comparison)
static void BM_malloc_churn(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<void*> ptrs;
    ptrs.reserve(state.range(0));

    // Allocate
    for (int64_t i = 0; i < state.range(0); ++i) {
      ptrs.push_back(std::malloc(64));
    }

    // Deallocate
    for (void* ptr : ptrs) {
      std::free(ptr);
    }
  }

  state.SetItemsProcessed(state.iterations() * state.range(0) * 2);
}
BENCHMARK(BM_malloc_churn)->Range(10, 1000)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Interleaved Allocation/Deallocation Pattern
// ============================================================================

// Benchmark: fixed_pool interleaved operations
static void BM_fixed_pool_interleaved(benchmark::State& state) {
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::fixed_pool<umpire::resource::host_memory<>>
    pool("bench_pool", &host, 128);

  for (auto _ : state) {
    std::vector<void*> ptrs;

    // Interleave allocations and deallocations
    for (int64_t i = 0; i < state.range(0); ++i) {
      ptrs.push_back(pool.allocate(128));

      if (i > 0 && i % 3 == 0 && !ptrs.empty()) {
        pool.deallocate(ptrs.back());
        ptrs.pop_back();
      }
    }

    // Cleanup remaining
    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_fixed_pool_interleaved)->Range(100, 1000)->Unit(benchmark::kMicrosecond);

// Benchmark: malloc/free interleaved operations (for comparison)
static void BM_malloc_interleaved(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<void*> ptrs;

    // Interleave allocations and deallocations
    for (int64_t i = 0; i < state.range(0); ++i) {
      ptrs.push_back(std::malloc(128));

      if (i > 0 && i % 3 == 0 && !ptrs.empty()) {
        std::free(ptrs.back());
        ptrs.pop_back();
      }
    }

    // Cleanup remaining
    for (void* ptr : ptrs) {
      std::free(ptr);
    }
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_malloc_interleaved)->Range(100, 1000)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Multiple Object Sizes
// ============================================================================

// Benchmark: fixed_pool with various object sizes
static void BM_fixed_pool_various_sizes(benchmark::State& state) {
  const std::size_t object_size = state.range(0);

  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::fixed_pool<umpire::resource::host_memory<>>
    pool("bench_pool", &host, object_size);

  for (auto _ : state) {
    void* ptr = pool.allocate(object_size);
    benchmark::DoNotOptimize(ptr);
    pool.deallocate(ptr);
  }

  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * object_size);
}
BENCHMARK(BM_fixed_pool_various_sizes)
  ->Arg(16)->Arg(32)->Arg(64)->Arg(128)->Arg(256)->Arg(512)->Arg(1024)->Arg(4096)
  ->Unit(benchmark::kNanosecond);

// ============================================================================
// Pool Growth Performance
// ============================================================================

// Benchmark: performance impact of pool growth
static void BM_fixed_pool_growth(benchmark::State& state) {
  auto& host = umpire::resource::host_memory<>::get();

  for (auto _ : state) {
    state.PauseTiming();
    umpire::strategy::fixed_pool<umpire::resource::host_memory<>>
      pool("bench_pool", &host, 64, 100);
    state.ResumeTiming();

    // Allocate enough to trigger multiple pool allocations
    std::vector<void*> ptrs;
    for (int64_t i = 0; i < state.range(0); ++i) {
      ptrs.push_back(pool.allocate(64));
    }

    state.PauseTiming();
    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
    state.ResumeTiming();
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_fixed_pool_growth)->Range(50, 500)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Many Small Allocations
// ============================================================================

// Benchmark: fixed_pool with many small allocations
static void BM_fixed_pool_many_small(benchmark::State& state) {
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::fixed_pool<umpire::resource::host_memory<>>
    pool("bench_pool", &host, 16, 5000);

  for (auto _ : state) {
    std::vector<void*> ptrs;
    ptrs.reserve(state.range(0));

    for (int64_t i = 0; i < state.range(0); ++i) {
      ptrs.push_back(pool.allocate(16));
    }

    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_fixed_pool_many_small)->Range(100, 5000)->Unit(benchmark::kMicrosecond);

// Benchmark: malloc with many small allocations (for comparison)
static void BM_malloc_many_small(benchmark::State& state) {
  for (auto _ : state) {
    std::vector<void*> ptrs;
    ptrs.reserve(state.range(0));

    for (int64_t i = 0; i < state.range(0); ++i) {
      ptrs.push_back(std::malloc(16));
    }

    for (void* ptr : ptrs) {
      std::free(ptr);
    }
  }

  state.SetItemsProcessed(state.iterations() * state.range(0));
}
BENCHMARK(BM_malloc_many_small)->Range(100, 5000)->Unit(benchmark::kMicrosecond);

// ============================================================================
// Release Performance
// ============================================================================

// Benchmark: fixed_pool release performance
static void BM_fixed_pool_release(benchmark::State& state) {
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::fixed_pool<umpire::resource::host_memory<>>
    pool("bench_pool", &host, 64, 100);

  for (auto _ : state) {
    state.PauseTiming();
    // Allocate and deallocate to create free pools
    std::vector<void*> ptrs;
    for (int i = 0; i < 500; ++i) {
      ptrs.push_back(pool.allocate(64));
    }
    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
    state.ResumeTiming();

    pool.release();
  }
}
BENCHMARK(BM_fixed_pool_release)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
