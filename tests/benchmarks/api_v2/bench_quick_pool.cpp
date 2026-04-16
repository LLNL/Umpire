//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/quick_pool.hpp"
#include "umpire/strategy/dynamic_pool_list.hpp"
#include "umpire/resource/host_memory.hpp"

#include <algorithm>
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
// Benchmark: O(1) Allocation Verification
// ============================================================================

void benchmark_o1_allocation() {
  std::cout << "\n=== Benchmark: O(1) Allocation Verification ===" << std::endl;

  bench_memory parent;
  umpire::strategy::quick_pool<bench_memory> pool("quick_pool", &parent);

  const int num_rounds = 1000;
  const int allocs_per_round = 100;

  Timer timer;
  timer.start();

  for (int round = 0; round < num_rounds; ++round) {
    std::vector<void*> ptrs;

    // Allocate
    for (int i = 0; i < allocs_per_round; ++i) {
      ptrs.push_back(pool.allocate(64));
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
// Benchmark: Comparison with dynamic_pool_list
// ============================================================================

void benchmark_vs_dynamic_pool() {
  std::cout << "\n=== Benchmark: quick_pool vs dynamic_pool_list ===" << std::endl;

  const int num_rounds = 1000;
  const int allocs_per_round = 100;

  // Test with different size ranges
  std::vector<std::pair<std::size_t, std::size_t>> size_ranges = {
    {16, 16},      // Single size
    {16, 256},     // Small range
    {16, 1024},    // Medium range
    {16, 4096},    // Full quick_pool range
  };

  for (const auto& range : size_ranges) {
    std::cout << "\n  Size range: " << range.first << " - " << range.second << " bytes" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> size_dist(range.first, range.second);

    // Benchmark quick_pool
    bench_memory parent1;
    umpire::strategy::quick_pool<bench_memory> quick("quick_pool", &parent1);

    Timer timer;
    timer.start();

    for (int round = 0; round < num_rounds; ++round) {
      std::vector<void*> ptrs;

      for (int i = 0; i < allocs_per_round; ++i) {
        std::size_t size = size_dist(gen);
        ptrs.push_back(quick.allocate(size));
      }

      for (void* ptr : ptrs) {
        quick.deallocate(ptr);
      }
    }

    double quick_time = timer.elapsed_ms();

    // Benchmark dynamic_pool_list
    bench_memory parent2;
    umpire::strategy::dynamic_pool_list<bench_memory> dynamic(
      "dynamic_pool", &parent2, 64 * 1024, 1024, 2.0);

    timer.start();

    for (int round = 0; round < num_rounds; ++round) {
      std::vector<void*> ptrs;

      for (int i = 0; i < allocs_per_round; ++i) {
        std::size_t size = size_dist(gen);
        ptrs.push_back(dynamic.allocate(size));
      }

      for (void* ptr : ptrs) {
        dynamic.deallocate(ptr);
      }
    }

    double dynamic_time = timer.elapsed_ms();

    std::cout << "    quick_pool time: " << quick_time << " ms" << std::endl;
    std::cout << "    dynamic_pool_list time: " << dynamic_time << " ms" << std::endl;
    std::cout << "    Speedup: " << std::fixed << std::setprecision(2)
              << (dynamic_time / quick_time) << "x" << std::endl;
  }
}

// ============================================================================
// Benchmark: Comparison with Direct malloc
// ============================================================================

void benchmark_vs_malloc() {
  std::cout << "\n=== Benchmark: quick_pool vs malloc ===" << std::endl;

  const int num_rounds = 1000;
  const int allocs_per_round = 100;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(16, 1024);

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

  // Benchmark quick_pool
  bench_memory parent;
  umpire::strategy::quick_pool<bench_memory> pool("quick_pool", &parent);

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
  std::cout << "  quick_pool time: " << pool_time << " ms" << std::endl;
  std::cout << "  Speedup: " << std::fixed << std::setprecision(2)
            << (malloc_time / pool_time) << "x" << std::endl;
}

// ============================================================================
// Benchmark: Various Allocation Sizes
// ============================================================================

void benchmark_size_classes() {
  std::cout << "\n=== Benchmark: Performance Across Size Classes ===" << std::endl;

  bench_memory parent;
  umpire::strategy::quick_pool<bench_memory> pool("quick_pool", &parent);

  const int num_rounds = 1000;
  const int allocs_per_round = 100;

  std::vector<std::size_t> test_sizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};

  for (std::size_t size : test_sizes) {
    Timer timer;
    timer.start();

    for (int round = 0; round < num_rounds; ++round) {
      std::vector<void*> ptrs;

      for (int i = 0; i < allocs_per_round; ++i) {
        ptrs.push_back(pool.allocate(size));
      }

      for (void* ptr : ptrs) {
        pool.deallocate(ptr);
      }
    }

    double elapsed = timer.elapsed_ms();
    double avg_us = (elapsed * 1000.0) / (num_rounds * allocs_per_round * 2.0);

    std::cout << "  Size " << std::setw(4) << size << " bytes: "
              << std::fixed << std::setprecision(3) << elapsed << " ms "
              << "(avg " << avg_us << " us/op)" << std::endl;
  }
}

// ============================================================================
// Benchmark: Internal Fragmentation Analysis
// ============================================================================

void benchmark_fragmentation() {
  std::cout << "\n=== Benchmark: Internal Fragmentation Analysis ===" << std::endl;

  using pool_type = umpire::strategy::quick_pool<bench_memory>;

  std::cout << "\n  Fragmentation by allocation size:" << std::endl;

  // Test various sizes and show fragmentation
  std::vector<std::size_t> test_sizes = {
    1, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 512, 1024, 2048, 4096, 4097
  };

  for (std::size_t size : test_sizes) {
    double frag = pool_type::calculate_fragmentation(size);
    std::size_t bin_index = 0;
    for (; bin_index < pool_type::get_num_bins(); ++bin_index) {
      if (size <= pool_type::get_bin_size(bin_index)) {
        break;
      }
    }

    if (bin_index < pool_type::get_num_bins()) {
      std::size_t bin_size = pool_type::get_bin_size(bin_index);
      std::size_t waste = bin_size - size;
      std::cout << "    Size " << std::setw(4) << size << " -> bin " << std::setw(4) << bin_size
                << " (waste: " << std::setw(4) << waste << " bytes, "
                << std::fixed << std::setprecision(1) << (frag * 100.0) << "%)" << std::endl;
    } else {
      std::cout << "    Size " << std::setw(4) << size << " -> direct allocation (no waste)" << std::endl;
    }
  }

  // Calculate average fragmentation for realistic workload
  bench_memory parent;
  umpire::strategy::quick_pool<bench_memory> pool("quick_pool", &parent);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(1, 4096);

  std::vector<std::size_t> sizes;
  for (int i = 0; i < 10000; ++i) {
    sizes.push_back(size_dist(gen));
  }

  double total_frag = 0.0;
  for (std::size_t size : sizes) {
    total_frag += pool_type::calculate_fragmentation(size);
  }

  std::cout << "\n  Average fragmentation (random 1-4096 bytes): "
            << std::fixed << std::setprecision(1)
            << (total_frag / sizes.size() * 100.0) << "%" << std::endl;
}

// ============================================================================
// Benchmark: Multi-Bin Stress Test
// ============================================================================

void benchmark_multi_bin_stress() {
  std::cout << "\n=== Benchmark: Multi-Bin Stress Test ===" << std::endl;

  bench_memory parent;
  umpire::strategy::quick_pool<bench_memory> pool("quick_pool", &parent);

  const int num_rounds = 100;
  const int allocs_per_round = 1000;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> size_dist(1, 4096);

  Timer timer;
  timer.start();

  for (int round = 0; round < num_rounds; ++round) {
    std::vector<void*> ptrs;

    // Allocate random sizes
    for (int i = 0; i < allocs_per_round; ++i) {
      std::size_t size = size_dist(gen);
      ptrs.push_back(pool.allocate(size));
    }

    // Deallocate in random order
    std::shuffle(ptrs.begin(), ptrs.end(), gen);

    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }

  double elapsed = timer.elapsed_ms();
  double ops_per_sec = (num_rounds * allocs_per_round * 2.0) / (elapsed / 1000.0);

  std::cout << "  Total time: " << elapsed << " ms" << std::endl;
  std::cout << "  Ops/sec: " << std::fixed << std::setprecision(0) << ops_per_sec << std::endl;

  std::cout << "\n  Final statistics:" << std::endl;
  std::cout << "    Total allocated: " << pool.get_total_allocated() << " bytes" << std::endl;
  std::cout << "    User allocated: " << pool.get_user_allocated() << " bytes" << std::endl;
  std::cout << "    Chunk count: " << pool.get_chunk_count() << std::endl;

  std::cout << "\n  Bin usage:" << std::endl;
  for (std::size_t i = 0; i < pool.get_num_bins(); ++i) {
    std::cout << "    Bin " << i << " (" << std::setw(4) << pool.get_bin_size(i) << " bytes): "
              << pool.get_bin_allocations(i) << " active, "
              << pool.get_bin_free_count(i) << " free" << std::endl;
  }
}

// ============================================================================
// Benchmark: Churn Pattern
// ============================================================================

void benchmark_churn_pattern() {
  std::cout << "\n=== Benchmark: Allocation Churn Pattern ===" << std::endl;

  bench_memory parent;
  umpire::strategy::quick_pool<bench_memory> pool("quick_pool", &parent);

  const int num_iterations = 10000;
  const int objects_per_iteration = 10;

  Timer timer;
  timer.start();

  for (int iter = 0; iter < num_iterations; ++iter) {
    std::vector<void*> ptrs;

    // Rapid allocation
    for (int i = 0; i < objects_per_iteration; ++i) {
      ptrs.push_back(pool.allocate(128));
    }

    // Immediate deallocation
    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }

  double elapsed = timer.elapsed_ms();
  double ops_per_sec = (num_iterations * objects_per_iteration * 2.0) / (elapsed / 1000.0);

  std::cout << "  Iterations: " << num_iterations << std::endl;
  std::cout << "  Objects per iteration: " << objects_per_iteration << std::endl;
  std::cout << "  Total time: " << elapsed << " ms" << std::endl;
  std::cout << "  Ops/sec: " << std::fixed << std::setprecision(0) << ops_per_sec << std::endl;
  std::cout << "  Chunk growth: " << pool.get_chunk_count() << " chunks" << std::endl;
}

// ============================================================================
// Benchmark: Large vs Small Allocations
// ============================================================================

void benchmark_large_vs_small() {
  std::cout << "\n=== Benchmark: Large vs Small Allocations ===" << std::endl;

  const int num_rounds = 1000;
  const int allocs_per_round = 100;

  // Small allocations (binned)
  {
    bench_memory parent;
    umpire::strategy::quick_pool<bench_memory> pool("quick_pool", &parent);

    Timer timer;
    timer.start();

    for (int round = 0; round < num_rounds; ++round) {
      std::vector<void*> ptrs;
      for (int i = 0; i < allocs_per_round; ++i) {
        ptrs.push_back(pool.allocate(64));
      }
      for (void* ptr : ptrs) {
        pool.deallocate(ptr);
      }
    }

    double small_time = timer.elapsed_ms();
    std::cout << "  Small allocations (64 bytes, binned): " << small_time << " ms" << std::endl;
  }

  // Large allocations (direct)
  {
    bench_memory parent;
    umpire::strategy::quick_pool<bench_memory> pool("quick_pool", &parent);

    Timer timer;
    timer.start();

    for (int round = 0; round < num_rounds; ++round) {
      std::vector<void*> ptrs;
      for (int i = 0; i < allocs_per_round; ++i) {
        ptrs.push_back(pool.allocate(8192));
      }
      for (void* ptr : ptrs) {
        pool.deallocate(ptr);
      }
    }

    double large_time = timer.elapsed_ms();
    std::cout << "  Large allocations (8192 bytes, direct): " << large_time << " ms" << std::endl;
  }
}

// ============================================================================
// Benchmark: Scalability Test
// ============================================================================

void benchmark_scalability() {
  std::cout << "\n=== Benchmark: Scalability Test ===" << std::endl;

  bench_memory parent;
  umpire::strategy::quick_pool<bench_memory> pool("quick_pool", &parent);

  std::vector<int> allocation_counts = {10, 100, 1000, 10000};

  for (int count : allocation_counts) {
    Timer timer;
    timer.start();

    std::vector<void*> ptrs;
    for (int i = 0; i < count; ++i) {
      ptrs.push_back(pool.allocate(128));
    }

    double alloc_time = timer.elapsed_us();

    timer.start();
    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }

    double dealloc_time = timer.elapsed_us();

    std::cout << "  " << std::setw(5) << count << " allocations: "
              << "alloc " << std::fixed << std::setprecision(2) << (alloc_time / count) << " us/op, "
              << "dealloc " << (dealloc_time / count) << " us/op" << std::endl;
  }
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "==================================================" << std::endl;
  std::cout << "  quick_pool Benchmark Suite" << std::endl;
  std::cout << "==================================================" << std::endl;

  benchmark_o1_allocation();
  benchmark_vs_dynamic_pool();
  benchmark_vs_malloc();
  benchmark_size_classes();
  benchmark_fragmentation();
  benchmark_multi_bin_stress();
  benchmark_churn_pattern();
  benchmark_large_vs_small();
  benchmark_scalability();

  std::cout << "\n==================================================" << std::endl;
  std::cout << "  Benchmarks Complete" << std::endl;
  std::cout << "==================================================" << std::endl;

  return 0;
}
