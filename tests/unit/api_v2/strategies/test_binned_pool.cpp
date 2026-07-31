//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/binned_pool.hpp"
#include "umpire/resource/host_memory.hpp"
#include "umpire/memory.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

// Test memory implementation for unit testing
class test_memory : public umpire::memory {
public:
  using platform = umpire::host_platform;

  test_memory() : umpire::memory{"test_parent"} { }

  void* allocate(std::size_t size) override
  {
    void* ptr{std::malloc(size)};
    track_allocation(ptr, size);
    return ptr;
  }

  void deallocate(void* ptr) override
  {
    untrack_allocation(ptr);
    std::free(ptr);
  }

  umpire::resource::Platform get_platform() const override {
    return umpire::resource::Platform::host;
  }
};

} // namespace

// ============================================================================
// Construction and Validation Tests
// ============================================================================

TEST(binned_pool, construct_with_valid_parameters)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  EXPECT_EQ(pool.get_parent(), &parent);
  EXPECT_EQ(pool.get_name(), "binned_pool");
  EXPECT_GT(pool.get_total_allocated(), 0);  // Pre-allocated chunks
  EXPECT_EQ(pool.get_user_allocated(), 0);
  EXPECT_GT(pool.get_chunk_count(), 0);
}

TEST(binned_pool, construct_with_nullptr_throws)
{
  EXPECT_THROW(
    umpire::strategy::binned_pool<test_memory> pool("binned_pool", nullptr),
    std::invalid_argument
  );
}

TEST(binned_pool, get_platform_delegates_to_parent)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  EXPECT_EQ(pool.get_platform(), parent.get_platform());
  EXPECT_EQ(pool.get_platform(), umpire::resource::Platform::host);
}

TEST(binned_pool, bin_configuration)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Verify bin sizes are power of 2
  EXPECT_EQ(umpire::strategy::binned_pool<test_memory>::get_num_bins(), 9);
  EXPECT_EQ(umpire::strategy::binned_pool<test_memory>::get_bin_size(0), 16);
  EXPECT_EQ(umpire::strategy::binned_pool<test_memory>::get_bin_size(1), 32);
  EXPECT_EQ(umpire::strategy::binned_pool<test_memory>::get_bin_size(2), 64);
  EXPECT_EQ(umpire::strategy::binned_pool<test_memory>::get_bin_size(3), 128);
  EXPECT_EQ(umpire::strategy::binned_pool<test_memory>::get_bin_size(4), 256);
  EXPECT_EQ(umpire::strategy::binned_pool<test_memory>::get_bin_size(5), 512);
  EXPECT_EQ(umpire::strategy::binned_pool<test_memory>::get_bin_size(6), 1024);
  EXPECT_EQ(umpire::strategy::binned_pool<test_memory>::get_bin_size(7), 2048);
  EXPECT_EQ(umpire::strategy::binned_pool<test_memory>::get_bin_size(8), 4096);

  EXPECT_EQ(pool.get_configured_bin_size(0), 16);
  EXPECT_EQ(pool.get_configured_bin_size(8), 4096);
  EXPECT_EQ(pool.get_blocks_per_bin(0), 1024);
  EXPECT_EQ(pool.get_blocks_per_bin(8), 8);
}

TEST(binned_pool, construct_with_custom_configuration)
{
  using pool_type = umpire::strategy::binned_pool<test_memory>;

  test_memory parent;
  pool_type::configuration_array bin_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  pool_type::configuration_array blocks_per_bin = {2, 2, 2, 2, 2, 1, 1, 1, 1};
  pool_type pool("binned_pool", &parent, bin_sizes, blocks_per_bin);

  EXPECT_EQ(pool.get_configured_bin_size(0), 32);
  EXPECT_EQ(pool.get_configured_bin_size(8), 8192);
  EXPECT_EQ(pool.get_blocks_per_bin(0), 2);
  EXPECT_EQ(pool.get_blocks_per_bin(8), 1);
  EXPECT_EQ(pool.get_bin_free_count(0), 2);
}

TEST(binned_pool, invalid_configuration_throws)
{
  using pool_type = umpire::strategy::binned_pool<test_memory>;

  test_memory parent;

  auto invalid_bins = pool_type::default_bin_sizes();
  auto blocks_per_bin = pool_type::default_blocks_per_bin();
  invalid_bins[3] = 96;
  EXPECT_THROW(pool_type("binned_pool", &parent, invalid_bins, blocks_per_bin),
               std::invalid_argument);

  auto non_monotonic_bins = pool_type::default_bin_sizes();
  non_monotonic_bins[4] = non_monotonic_bins[3];
  EXPECT_THROW(pool_type("binned_pool", &parent, non_monotonic_bins, blocks_per_bin),
               std::invalid_argument);

  auto invalid_blocks = pool_type::default_blocks_per_bin();
  invalid_blocks[0] = 0;
  EXPECT_THROW(pool_type("binned_pool", &parent, pool_type::default_bin_sizes(), invalid_blocks),
               std::invalid_argument);
}

// ============================================================================
// Basic Allocation Tests
// ============================================================================

TEST(binned_pool, basic_allocation)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  void* ptr = pool.allocate(64);
  EXPECT_NE(ptr, nullptr);

  // Verify statistics updated
  EXPECT_GT(pool.get_user_allocated(), 0);

  pool.deallocate(ptr);
  EXPECT_EQ(pool.get_user_allocated(), 0);
}

TEST(binned_pool, zero_size_allocation_returns_nullptr)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  void* ptr = pool.allocate(0);
  EXPECT_EQ(ptr, nullptr);
}

TEST(binned_pool, multiple_allocations)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  std::vector<void*> ptrs;
  for (int i = 0; i < 10; ++i) {
    void* ptr = pool.allocate(64);
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  EXPECT_GT(pool.get_user_allocated(), 0);

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }

  EXPECT_EQ(pool.get_user_allocated(), 0);
}

TEST(binned_pool, nullptr_deallocation_is_safe)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  std::size_t allocated_before = pool.get_user_allocated();
  EXPECT_NO_THROW(pool.deallocate(nullptr));
  EXPECT_EQ(pool.get_user_allocated(), allocated_before);
}

// ============================================================================
// Power-of-2 Bin Selection Tests
// ============================================================================

TEST(binned_pool, bin_selection_exact_sizes)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Exact bin sizes should map correctly
  std::vector<std::size_t> sizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
  std::vector<void*> ptrs;

  for (std::size_t size : sizes) {
    void* ptr = pool.allocate(size);
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }
}

TEST(binned_pool, bin_selection_rounding_up)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Test that sizes round up to next power-of-2
  struct test_case {
    std::size_t request_size;
    std::size_t expected_bin_index;
    std::size_t expected_bin_size;
  };

  std::vector<test_case> tests = {
    {1, 0, 16},       // Rounds up to 16
    {17, 1, 32},      // Rounds up to 32
    {33, 2, 64},      // Rounds up to 64
    {65, 3, 128},     // Rounds up to 128
    {100, 3, 128},    // Rounds up to 128
    {129, 4, 256},    // Rounds up to 256
    {1000, 6, 1024},  // Rounds up to 1024
    {2049, 8, 4096},  // Rounds up to 4096
  };

  for (const auto& test : tests) {
    EXPECT_EQ(umpire::strategy::binned_pool<test_memory>::get_bin_size(test.expected_bin_index),
              test.expected_bin_size);
    EXPECT_EQ(pool.get_bin_allocations(test.expected_bin_index), 0);

    void* ptr = pool.allocate(test.request_size);
    EXPECT_NE(ptr, nullptr);

    EXPECT_EQ(pool.get_bin_allocations(test.expected_bin_index), 1);
    pool.deallocate(ptr);
    EXPECT_EQ(pool.get_bin_allocations(test.expected_bin_index), 0);
  }
}

TEST(binned_pool, small_allocation_uses_correct_bin)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Allocate 1 byte - should use bin 0 (16 bytes)
  void* ptr = pool.allocate(1);
  EXPECT_NE(ptr, nullptr);

  // Bin 0 should have one allocation
  EXPECT_EQ(pool.get_bin_allocations(0), 1);

  pool.deallocate(ptr);
  EXPECT_EQ(pool.get_bin_allocations(0), 0);
}

// ============================================================================
// Large Allocation Tests
// ============================================================================

TEST(binned_pool, large_allocation_forwarded_to_parent)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Allocations > 4096 should be forwarded to parent
  void* ptr = pool.allocate(8192);
  EXPECT_NE(ptr, nullptr);

  pool.deallocate(ptr);
}

TEST(binned_pool, very_large_allocation)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  void* ptr = pool.allocate(1024 * 1024);
  EXPECT_NE(ptr, nullptr);

  pool.deallocate(ptr);
}

TEST(binned_pool, mixed_small_and_large_allocations)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  std::vector<void*> ptrs;

  // Mix of small (binned) and large (direct) allocations
  ptrs.push_back(pool.allocate(64));      // Binned
  ptrs.push_back(pool.allocate(8192));    // Direct
  ptrs.push_back(pool.allocate(128));     // Binned
  ptrs.push_back(pool.allocate(16384));   // Direct
  ptrs.push_back(pool.allocate(512));     // Binned

  for (void* ptr : ptrs) {
    EXPECT_NE(ptr, nullptr);
  }

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }
}

// ============================================================================
// Internal Fragmentation Tests
// ============================================================================

TEST(binned_pool, internal_fragmentation_measurement)
{
  // Test fragmentation calculation
  using pool_type = umpire::strategy::binned_pool<test_memory>;

  // 1 byte in 16-byte bin: 93.75% fragmentation
  double frag1 = pool_type::calculate_fragmentation(1);
  EXPECT_NEAR(frag1, 0.9375, 0.01);

  // 65 bytes in 128-byte bin: ~49% fragmentation
  double frag65 = pool_type::calculate_fragmentation(65);
  EXPECT_NEAR(frag65, 0.49, 0.01);

  // 128 bytes in 128-byte bin: 0% fragmentation
  double frag128 = pool_type::calculate_fragmentation(128);
  EXPECT_NEAR(frag128, 0.0, 0.01);

  // Large allocation: 0% fragmentation (direct)
  double frag_large = pool_type::calculate_fragmentation(8192);
  EXPECT_EQ(frag_large, 0.0);
}

TEST(binned_pool, worst_case_fragmentation)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Allocate sizes just over bin boundary (worst fragmentation)
  std::vector<void*> ptrs;
  ptrs.push_back(pool.allocate(17));   // Uses 32-byte bin (47% waste)
  ptrs.push_back(pool.allocate(33));   // Uses 64-byte bin (48% waste)
  ptrs.push_back(pool.allocate(65));   // Uses 128-byte bin (49% waste)
  ptrs.push_back(pool.allocate(129));  // Uses 256-byte bin (50% waste)

  for (void* ptr : ptrs) {
    EXPECT_NE(ptr, nullptr);
    pool.deallocate(ptr);
  }
}

// ============================================================================
// Free List Operations Tests
// ============================================================================

TEST(binned_pool, free_list_reuse)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Allocate and free multiple times
  void* ptr1 = pool.allocate(64);
  pool.deallocate(ptr1);

  void* ptr2 = pool.allocate(64);
  pool.deallocate(ptr2);

  void* ptr3 = pool.allocate(64);
  pool.deallocate(ptr3);

  // Should not need to allocate new chunks (reuses from free list)
  std::size_t chunk_count = pool.get_chunk_count();
  EXPECT_GT(chunk_count, 0);

  // Allocate many more - should still use existing chunks
  for (int i = 0; i < 10; ++i) {
    void* ptr = pool.allocate(64);
    pool.deallocate(ptr);
  }

  // Chunk count shouldn't increase significantly
  EXPECT_LE(pool.get_chunk_count(), chunk_count + 1);
}

TEST(binned_pool, multiple_bins_free_lists)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Allocate from different bins
  void* ptr16 = pool.allocate(16);
  void* ptr32 = pool.allocate(32);
  void* ptr64 = pool.allocate(64);
  void* ptr128 = pool.allocate(128);

  EXPECT_EQ(pool.get_bin_allocations(0), 1);  // 16-byte bin
  EXPECT_EQ(pool.get_bin_allocations(1), 1);  // 32-byte bin
  EXPECT_EQ(pool.get_bin_allocations(2), 1);  // 64-byte bin
  EXPECT_EQ(pool.get_bin_allocations(3), 1);  // 128-byte bin

  pool.deallocate(ptr16);
  pool.deallocate(ptr32);
  pool.deallocate(ptr64);
  pool.deallocate(ptr128);

  EXPECT_EQ(pool.get_bin_allocations(0), 0);
  EXPECT_EQ(pool.get_bin_allocations(1), 0);
  EXPECT_EQ(pool.get_bin_allocations(2), 0);
  EXPECT_EQ(pool.get_bin_allocations(3), 0);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(binned_pool, statistics_accuracy)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  EXPECT_EQ(pool.get_user_allocated(), 0);

  // Allocate one rounded request and one direct request.
  void* ptr1 = pool.allocate(65);
  void* ptr2 = pool.allocate(8192);

  EXPECT_EQ(pool.get_user_allocated(), 65 + 8192);

  // Deallocate one
  pool.deallocate(ptr1);
  EXPECT_EQ(pool.get_user_allocated(), 8192);

  // Deallocate all
  pool.deallocate(ptr2);
  EXPECT_EQ(pool.get_user_allocated(), 0);
}

TEST(binned_pool, direct_allocation_returns_total_to_baseline)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  const std::size_t baseline_total = pool.get_total_allocated();

  void* ptr = pool.allocate(8192);
  ASSERT_NE(ptr, nullptr);
  EXPECT_GT(pool.get_total_allocated(), baseline_total);

  pool.deallocate(ptr);
  EXPECT_EQ(pool.get_total_allocated(), baseline_total);
}

TEST(binned_pool, bin_statistics)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Allocate multiple from same bin
  std::vector<void*> ptrs;
  for (int i = 0; i < 5; ++i) {
    ptrs.push_back(pool.allocate(64));
  }

  EXPECT_EQ(pool.get_bin_allocations(2), 5);  // 64-byte bin

  // Deallocate some
  pool.deallocate(ptrs[0]);
  pool.deallocate(ptrs[1]);

  EXPECT_EQ(pool.get_bin_allocations(2), 3);

  // Cleanup
  for (std::size_t i = 2; i < ptrs.size(); ++i) {
    pool.deallocate(ptrs[i]);
  }
}

TEST(binned_pool, free_count_tracking)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Get initial free count for bin 2 (64 bytes)
  std::size_t initial_free = pool.get_bin_free_count(2);
  EXPECT_GT(initial_free, 0);  // Should have pre-allocated objects

  // Allocate some
  void* ptr1 = pool.allocate(64);
  void* ptr2 = pool.allocate(64);

  EXPECT_EQ(pool.get_bin_free_count(2), initial_free - 2);

  // Deallocate
  pool.deallocate(ptr1);
  pool.deallocate(ptr2);

  EXPECT_EQ(pool.get_bin_free_count(2), initial_free);
}

// ============================================================================
// Memory Content Tests
// ============================================================================

TEST(binned_pool, allocated_memory_is_writable)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  void* ptr = pool.allocate(256);
  ASSERT_NE(ptr, nullptr);

  // Write to memory
  char* bytes = static_cast<char*>(ptr);
  for (int i = 0; i < 256; ++i) {
    bytes[i] = static_cast<char>(i & 0xFF);
  }

  // Verify written values
  for (int i = 0; i < 256; ++i) {
    EXPECT_EQ(bytes[i], static_cast<char>(i & 0xFF));
  }

  pool.deallocate(ptr);
}

TEST(binned_pool, allocation_alignment_is_preserved)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  void* small = pool.allocate(1);
  ASSERT_NE(small, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(small) % alignof(std::max_align_t), 0u);
  pool.deallocate(small);

  void* large = pool.allocate(8192);
  ASSERT_NE(large, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(large) % alignof(std::max_align_t), 0u);
  pool.deallocate(large);
}

TEST(binned_pool, unique_allocations)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Allocate multiple blocks and verify they're unique
  std::vector<void*> ptrs;
  for (int i = 0; i < 20; ++i) {
    void* ptr = pool.allocate(128);
    EXPECT_NE(ptr, nullptr);

    // Verify this pointer is unique
    for (void* existing : ptrs) {
      EXPECT_NE(ptr, existing);
    }

    ptrs.push_back(ptr);
  }

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }
}

// ============================================================================
// Allocation Pattern Tests
// ============================================================================

TEST(binned_pool, churn_pattern)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Simulate allocation churn
  for (int iter = 0; iter < 100; ++iter) {
    std::vector<void*> ptrs;
    for (int i = 0; i < 20; ++i) {
      ptrs.push_back(pool.allocate(128));
    }

    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }

  // Should still be functional
  EXPECT_EQ(pool.get_user_allocated(), 0);
}

TEST(binned_pool, interleaved_alloc_dealloc)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  std::vector<void*> ptrs;

  // Interleave allocations and deallocations
  for (int i = 0; i < 30; ++i) {
    ptrs.push_back(pool.allocate(256));

    if (i > 0 && i % 3 == 0 && !ptrs.empty()) {
      pool.deallocate(ptrs.back());
      ptrs.pop_back();
    }
  }

  // Cleanup
  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }

  EXPECT_EQ(pool.get_user_allocated(), 0);
}

TEST(binned_pool, custom_blocks_per_bin_control_growth)
{
  using pool_type = umpire::strategy::binned_pool<test_memory>;

  test_memory parent;
  auto bin_sizes = pool_type::default_bin_sizes();
  auto blocks_per_bin = pool_type::default_blocks_per_bin();
  blocks_per_bin[0] = 2;

  pool_type pool("binned_pool", &parent, bin_sizes, blocks_per_bin);
  const std::size_t initial_chunk_count = pool.get_chunk_count();

  void* ptr1 = pool.allocate(16);
  void* ptr2 = pool.allocate(16);
  EXPECT_EQ(pool.get_bin_free_count(0), 0);

  void* ptr3 = pool.allocate(16);
  EXPECT_EQ(pool.get_chunk_count(), initial_chunk_count + 1);

  pool.deallocate(ptr1);
  pool.deallocate(ptr2);
  pool.deallocate(ptr3);
}

TEST(binned_pool, variable_sized_allocations)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  std::vector<std::size_t> sizes = {16, 32, 64, 128, 256, 512, 1024, 64, 32};
  std::vector<void*> ptrs;

  for (std::size_t size : sizes) {
    void* ptr = pool.allocate(size);
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Verify all unique
  for (size_t i = 0; i < ptrs.size(); ++i) {
    for (size_t j = i + 1; j < ptrs.size(); ++j) {
      EXPECT_NE(ptrs[i], ptrs[j]);
    }
  }

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }
}

// ============================================================================
// Platform Type Propagation Tests
// ============================================================================

TEST(binned_pool, platform_type_propagation)
{
  using host_mem = umpire::resource::host_memory<>;
  using binned_pool_host = umpire::strategy::binned_pool<host_mem>;

  // Platform type should be propagated from host_memory
  static_assert(std::is_same<binned_pool_host::platform, umpire::host_platform>::value,
                "Platform type should be propagated from wrapped memory");
}

// ============================================================================
// Composition Tests
// ============================================================================

TEST(binned_pool, composition_with_host_memory)
{
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::binned_pool<umpire::resource::host_memory<>>
    pool("binned_pool_host", &host);

  void* ptr = pool.allocate(512);
  EXPECT_NE(ptr, nullptr);

  // Write to verify memory is accessible
  char* bytes = static_cast<char*>(ptr);
  bytes[0] = 'A';
  bytes[511] = 'Z';
  EXPECT_EQ(bytes[0], 'A');
  EXPECT_EQ(bytes[511], 'Z');

  pool.deallocate(ptr);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(binned_pool, boundary_sizes)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Test sizes at bin boundaries
  std::vector<std::size_t> boundary_sizes = {
    4095, 4096, 4097  // Around max bin size
  };

  for (std::size_t size : boundary_sizes) {
    void* ptr = pool.allocate(size);
    EXPECT_NE(ptr, nullptr);
    pool.deallocate(ptr);
  }
}

TEST(binned_pool, many_small_allocations)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  std::vector<void*> ptrs;
  for (int i = 0; i < 1000; ++i) {
    ptrs.push_back(pool.allocate(16));
  }

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }

  EXPECT_EQ(pool.get_user_allocated(), 0);
}

TEST(binned_pool, all_bins_used)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  // Allocate from every bin
  std::vector<void*> ptrs;
  for (std::size_t i = 0; i < pool.get_num_bins(); ++i) {
    std::size_t size = pool.get_bin_size(i);
    void* ptr = pool.allocate(size);
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);

    EXPECT_EQ(pool.get_bin_allocations(i), 1);
  }

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }
}

TEST(binned_pool, release_is_safe)
{
  test_memory parent;
  umpire::strategy::binned_pool<test_memory> pool("binned_pool", &parent);

  void* ptr = pool.allocate(128);
  pool.deallocate(ptr);

  // Release should be safe (even if it's a no-op)
  EXPECT_NO_THROW(pool.release());

  // Should still be able to allocate after release
  void* ptr2 = pool.allocate(128);
  EXPECT_NE(ptr2, nullptr);
  pool.deallocate(ptr2);
}
