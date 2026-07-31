//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/coalescing_pool_list.hpp"
#include "umpire/resource/host_memory.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

// Test memory implementation for benchmarking
class bench_memory : public umpire::memory {
public:
  using platform = umpire::host_platform;

  bench_memory() : umpire::memory{"bench_parent"} { }

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

// Timer helper class
class Timer {
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;

public:
  void start() {
    start_ = std::chrono::high_resolution_clock::now();
  }

  double elapsed_us() const {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
    return static_cast<double>(duration.count());
  }

  double elapsed_ms() const {
    return elapsed_us() / 1000.0;
  }
};

} // namespace

// ============================================================================
// Benchmark: Random Size Allocations
// ============================================================================

void benchmark_random_size_allocations() {
  std::cout << "\n=== Benchmark: Random Size Allocations ===" << std::endl;

  bench_memory parent;
  umpire::strategy::coalescing_pool_list<bench_memory> pool(
    "dynamic_pool", &parent, 1024 * 1024, 4096, 2.0);

  const int num_rounds = 1000;
  const int allocs_per_round = 100;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(512, 8192);

  Timer timer;
  timer.start();

  for (int round = 0; round < num_rounds; ++round) {
    std::vector<void*> ptrs;

    // Allocate
    for (int i = 0; i < allocs_per_round; ++i) {
      std::size_t size = size_dist(gen);
      ptrs.push_back(pool.allocate(size));
    }

    // Deallocate
    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }

  double elapsed = timer.elapsed_ms();
  double ops_per_sec = (num_rounds * allocs_per_round * 2.0) / (elapsed / 1000.0);

  std::cout << "  Rounds: " << num_rounds << std::endl;
  std::cout << "  Allocations per round: " << allocs_per_round << std::endl;
  std::cout << "  Total time: " << elapsed << " ms" << std::endl;
  std::cout << "  Ops/sec: " << std::fixed << std::setprecision(0) << ops_per_sec << std::endl;
  std::cout << "  Avg time per op: " << std::fixed << std::setprecision(3)
            << (elapsed * 1000.0) / (num_rounds * allocs_per_round * 2.0) << " us" << std::endl;
}

// ============================================================================
// Benchmark: Fragmentation Patterns
// ============================================================================

void benchmark_fragmentation_patterns() {
  std::cout << "\n=== Benchmark: Fragmentation Patterns ===" << std::endl;

  bench_memory parent;
  umpire::strategy::coalescing_pool_list<bench_memory> pool(
    "dynamic_pool", &parent, 512 * 1024, 2048, 2.0);

  const int num_rounds = 100;
  const int num_allocs = 500;

  Timer timer;

  // Pattern 1: Sequential allocation/deallocation
  timer.start();
  for (int round = 0; round < num_rounds; ++round) {
    std::vector<void*> ptrs;
    for (int i = 0; i < num_allocs; ++i) {
      ptrs.push_back(pool.allocate(1024));
    }
    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }
  double sequential_time = timer.elapsed_ms();

  std::cout << "  Sequential pattern:" << std::endl;
  std::cout << "    Time: " << sequential_time << " ms" << std::endl;

  // Pattern 2: Reverse order deallocation
  timer.start();
  for (int round = 0; round < num_rounds; ++round) {
    std::vector<void*> ptrs;
    for (int i = 0; i < num_allocs; ++i) {
      ptrs.push_back(pool.allocate(1024));
    }
    for (auto it = ptrs.rbegin(); it != ptrs.rend(); ++it) {
      pool.deallocate(*it);
    }
  }
  double reverse_time = timer.elapsed_ms();

  std::cout << "  Reverse pattern:" << std::endl;
  std::cout << "    Time: " << reverse_time << " ms" << std::endl;

  // Pattern 3: Alternating deallocation (high fragmentation)
  timer.start();
  for (int round = 0; round < num_rounds; ++round) {
    std::vector<void*> ptrs;
    for (int i = 0; i < num_allocs; ++i) {
      ptrs.push_back(pool.allocate(1024));
    }

    // Deallocate every other block
    for (size_t i = 0; i < ptrs.size(); i += 2) {
      pool.deallocate(ptrs[i]);
    }

    // Deallocate remaining
    for (size_t i = 1; i < ptrs.size(); i += 2) {
      pool.deallocate(ptrs[i]);
    }
  }
  double alternating_time = timer.elapsed_ms();

  std::cout << "  Alternating pattern:" << std::endl;
  std::cout << "    Time: " << alternating_time << " ms" << std::endl;

  // Pattern 4: Random deallocation
  std::random_device rd;
  std::mt19937 gen(rd());

  timer.start();
  for (int round = 0; round < num_rounds; ++round) {
    std::vector<void*> ptrs;
    for (int i = 0; i < num_allocs; ++i) {
      ptrs.push_back(pool.allocate(1024));
    }

    std::shuffle(ptrs.begin(), ptrs.end(), gen);

    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }
  double random_time = timer.elapsed_ms();

  std::cout << "  Random pattern:" << std::endl;
  std::cout << "    Time: " << random_time << " ms" << std::endl;
}

// ============================================================================
// Benchmark: Comparison with malloc
// ============================================================================

void benchmark_malloc_comparison() {
  std::cout << "\n=== Benchmark: Comparison with malloc ===" << std::endl;

  const int num_rounds = 1000;
  const int allocs_per_round = 100;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(512, 8192);

  // Benchmark malloc
  Timer timer;
  timer.start();

  for (int round = 0; round < num_rounds; ++round) {
    std::vector<void*> ptrs;

    for (int i = 0; i < allocs_per_round; ++i) {
      std::size_t size = size_dist(gen);
      ptrs.push_back(std::malloc(size));
    }

    for (void* ptr : ptrs) {
      std::free(ptr);
    }
  }

  double malloc_time = timer.elapsed_ms();

  // Benchmark coalescing_pool_list
  bench_memory parent;
  umpire::strategy::coalescing_pool_list<bench_memory> pool(
    "dynamic_pool", &parent, 1024 * 1024, 4096, 2.0);

  timer.start();

  for (int round = 0; round < num_rounds; ++round) {
    std::vector<void*> ptrs;

    for (int i = 0; i < allocs_per_round; ++i) {
      std::size_t size = size_dist(gen);
      ptrs.push_back(pool.allocate(size));
    }

    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }

  double pool_time = timer.elapsed_ms();

  std::cout << "  malloc time: " << malloc_time << " ms" << std::endl;
  std::cout << "  coalescing_pool_list time: " << pool_time << " ms" << std::endl;
  std::cout << "  Speedup: " << std::fixed << std::setprecision(2)
            << (malloc_time / pool_time) << "x" << std::endl;
}

// ============================================================================
// Benchmark: Coalescing Overhead
// ============================================================================

void benchmark_coalescing_overhead() {
  std::cout << "\n=== Benchmark: Coalescing Overhead ===" << std::endl;

  bench_memory parent;
  umpire::strategy::coalescing_pool_list<bench_memory> pool(
    "dynamic_pool", &parent, 1024 * 1024, 1024, 2.0);

  const int num_rounds = 100;
  const int num_blocks = 1000;

  Timer timer;

  // Scenario 1: No coalescing (allocate and hold)
  timer.start();
  for (int round = 0; round < num_rounds; ++round) {
    std::vector<void*> ptrs;
    for (int i = 0; i < num_blocks; ++i) {
      ptrs.push_back(pool.allocate(2048));
    }

    // Deallocate in order (minimal coalescing)
    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }
  double no_coalesce_time = timer.elapsed_ms();

  std::cout << "  Sequential deallocation (forward): "
            << no_coalesce_time << " ms" << std::endl;

  // Scenario 2: Maximum coalescing (deallocate in reverse)
  timer.start();
  for (int round = 0; round < num_rounds; ++round) {
    std::vector<void*> ptrs;
    for (int i = 0; i < num_blocks; ++i) {
      ptrs.push_back(pool.allocate(2048));
    }

    // Deallocate in reverse (maximum coalescing)
    for (auto it = ptrs.rbegin(); it != ptrs.rend(); ++it) {
      pool.deallocate(*it);
    }
  }
  double max_coalesce_time = timer.elapsed_ms();

  std::cout << "  Sequential deallocation (reverse): "
            << max_coalesce_time << " ms" << std::endl;

  // Scenario 3: Alternating pattern (medium coalescing)
  timer.start();
  for (int round = 0; round < num_rounds; ++round) {
    std::vector<void*> ptrs;
    for (int i = 0; i < num_blocks; ++i) {
      ptrs.push_back(pool.allocate(2048));
    }

    // Deallocate every other, then the rest
    for (size_t i = 1; i < ptrs.size(); i += 2) {
      pool.deallocate(ptrs[i]);
    }
    for (size_t i = 0; i < ptrs.size(); i += 2) {
      pool.deallocate(ptrs[i]);
    }
  }
  double medium_coalesce_time = timer.elapsed_ms();

  std::cout << "  Alternating deallocation: "
            << medium_coalesce_time << " ms" << std::endl;

  std::cout << "  Coalescing overhead (forward vs reverse): "
            << std::fixed << std::setprecision(2)
            << ((max_coalesce_time - no_coalesce_time) / no_coalesce_time * 100.0)
            << "%" << std::endl;
}

// ============================================================================
// Benchmark: Variable Size Stress Test
// ============================================================================

void benchmark_variable_size_stress() {
  std::cout << "\n=== Benchmark: Variable Size Stress Test ===" << std::endl;

  bench_memory parent;
  umpire::strategy::coalescing_pool_list<bench_memory> pool(
    "dynamic_pool", &parent, 2 * 1024 * 1024, 4096, 1.5);

  const int num_rounds = 50;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(256, 65536);  // Wide range
  std::uniform_int_distribution<> lifetime_dist(1, 50);

  Timer timer;
  timer.start();

  std::vector<std::pair<void*, int>> active_allocs;

  for (int round = 0; round < num_rounds * 100; ++round) {
    // Allocate new blocks
    for (int i = 0; i < 10; ++i) {
      std::size_t size = size_dist(gen);
      int lifetime = lifetime_dist(gen);
      void* ptr = pool.allocate(size);
      active_allocs.push_back({ptr, lifetime});
    }

    // Age and deallocate old blocks
    for (auto it = active_allocs.begin(); it != active_allocs.end(); ) {
      it->second--;
      if (it->second <= 0) {
        pool.deallocate(it->first);
        it = active_allocs.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Cleanup remaining
  for (auto& alloc : active_allocs) {
    pool.deallocate(alloc.first);
  }

  double elapsed = timer.elapsed_ms();

  std::cout << "  Total time: " << elapsed << " ms" << std::endl;
  std::cout << "  Final stats:" << std::endl;
  std::cout << "    Total size: " << pool.get_total_size() << " bytes" << std::endl;
  std::cout << "    Allocated: " << pool.get_allocated_size() << " bytes" << std::endl;
  std::cout << "    Free: " << pool.get_free_size() << " bytes" << std::endl;
  std::cout << "    Block count: " << pool.get_block_count() << std::endl;
}

// ============================================================================
// Benchmark: Pool Growth Performance
// ============================================================================

void benchmark_pool_growth() {
  std::cout << "\n=== Benchmark: Pool Growth Performance ===" << std::endl;

  bench_memory parent;

  // Test with different growth factors
  std::vector<double> growth_factors = {1.5, 2.0, 3.0};

  for (double growth_factor : growth_factors) {
    umpire::strategy::coalescing_pool_list<bench_memory> pool(
      "dynamic_pool", &parent, 64 * 1024, 4096, growth_factor);

    Timer timer;
    timer.start();

    std::vector<void*> ptrs;
    // Allocate until we've grown the pool multiple times
    for (int i = 0; i < 1000; ++i) {
      ptrs.push_back(pool.allocate(4096));
    }

    double alloc_time = timer.elapsed_ms();

    // Deallocate
    timer.start();
    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
    double dealloc_time = timer.elapsed_ms();

    std::cout << "  Growth factor " << growth_factor << ":" << std::endl;
    std::cout << "    Allocation time: " << alloc_time << " ms" << std::endl;
    std::cout << "    Deallocation time: " << dealloc_time << " ms" << std::endl;
    std::cout << "    Final total size: " << pool.get_total_size() << " bytes" << std::endl;
  }
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "==================================================" << std::endl;
  std::cout << "  coalescing_pool_list Benchmark Suite" << std::endl;
  std::cout << "==================================================" << std::endl;

  benchmark_random_size_allocations();
  benchmark_fragmentation_patterns();
  benchmark_malloc_comparison();
  benchmark_coalescing_overhead();
  benchmark_variable_size_stress();
  benchmark_pool_growth();

  std::cout << "\n==================================================" << std::endl;
  std::cout << "  Benchmarks Complete" << std::endl;
  std::cout << "==================================================" << std::endl;

  return 0;
}
