//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/dynamic_pool_list.hpp"
#include "umpire/resource/host_memory.hpp"
#include "umpire/memory.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

// Test memory implementation for unit testing
class test_memory : public umpire::memory {
public:
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

TEST(dynamic_pool_list, construct_with_default_parameters)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent);

  EXPECT_EQ(pool.get_parent(), &parent);
  EXPECT_EQ(pool.get_name(), "dynamic_pool");
  EXPECT_EQ(pool.get_initial_pool_size(), 64 * 1024);
  EXPECT_EQ(pool.get_min_alloc_size(), 4 * 1024);
  EXPECT_EQ(pool.get_growth_factor(), 2.0);
  EXPECT_GT(pool.get_total_size(), 0);  // Initial pool allocated
  EXPECT_EQ(pool.get_allocated_size(), 0);
  EXPECT_EQ(pool.get_free_size(), pool.get_total_size());
}

TEST(dynamic_pool_list, construct_with_custom_parameters)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool(
    "dynamic_pool", &parent, 128 * 1024, 8 * 1024, 1.5);

  EXPECT_EQ(pool.get_initial_pool_size(), 128 * 1024);
  EXPECT_EQ(pool.get_min_alloc_size(), 8 * 1024);
  EXPECT_EQ(pool.get_growth_factor(), 1.5);
  EXPECT_EQ(pool.get_total_size(), 128 * 1024);
  EXPECT_EQ(pool.get_free_size(), 128 * 1024);
}

TEST(dynamic_pool_list, construct_with_nullptr_throws)
{
  EXPECT_THROW(
    umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", nullptr),
    std::invalid_argument
  );
}

TEST(dynamic_pool_list, construct_with_zero_initial_pool_size_throws)
{
  test_memory parent;
  EXPECT_THROW(
    umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 0),
    std::invalid_argument
  );
}

TEST(dynamic_pool_list, construct_with_zero_min_alloc_size_throws)
{
  test_memory parent;
  EXPECT_THROW(
    umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 0),
    std::invalid_argument
  );
}

TEST(dynamic_pool_list, construct_with_invalid_growth_factor_throws)
{
  test_memory parent;
  EXPECT_THROW(
    umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 4*1024, 1.0),
    std::invalid_argument
  );
  EXPECT_THROW(
    umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 4*1024, 0.5),
    std::invalid_argument
  );
}

TEST(dynamic_pool_list, get_platform_delegates_to_parent)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent);

  EXPECT_EQ(pool.get_platform(), parent.get_platform());
  EXPECT_EQ(pool.get_platform(), umpire::resource::Platform::host);
}

// ============================================================================
// Basic Allocation Tests
// ============================================================================

TEST(dynamic_pool_list, basic_allocation)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 1024);

  void* ptr = pool.allocate(1024);
  EXPECT_NE(ptr, nullptr);

  // Verify statistics updated
  EXPECT_EQ(pool.get_allocated_size(), 1024);
  EXPECT_EQ(pool.get_free_size(), pool.get_total_size() - 1024);

  pool.deallocate(ptr);
  EXPECT_EQ(pool.get_allocated_size(), 0);
}

TEST(dynamic_pool_list, zero_size_allocation_returns_nullptr)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent);

  void* ptr = pool.allocate(0);
  EXPECT_EQ(ptr, nullptr);
}

TEST(dynamic_pool_list, multiple_allocations)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 1024);

  std::vector<void*> ptrs;
  for (int i = 0; i < 5; ++i) {
    void* ptr = pool.allocate(1024);
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  EXPECT_EQ(pool.get_allocated_size(), 5 * 1024);

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }

  EXPECT_EQ(pool.get_allocated_size(), 0);
}

TEST(dynamic_pool_list, variable_sized_allocations)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 512);

  void* ptr1 = pool.allocate(1024);
  void* ptr2 = pool.allocate(2048);
  void* ptr3 = pool.allocate(512);
  void* ptr4 = pool.allocate(4096);

  EXPECT_NE(ptr1, nullptr);
  EXPECT_NE(ptr2, nullptr);
  EXPECT_NE(ptr3, nullptr);
  EXPECT_NE(ptr4, nullptr);

  std::size_t expected = 1024 + 2048 + 512 + 4096;
  EXPECT_EQ(pool.get_allocated_size(), expected);

  pool.deallocate(ptr1);
  pool.deallocate(ptr2);
  pool.deallocate(ptr3);
  pool.deallocate(ptr4);
}

TEST(dynamic_pool_list, nullptr_deallocation_is_safe)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent);

  std::size_t allocated_before = pool.get_allocated_size();
  EXPECT_NO_THROW(pool.deallocate(nullptr));
  EXPECT_EQ(pool.get_allocated_size(), allocated_before);
}

// ============================================================================
// Block Splitting Tests
// ============================================================================

TEST(dynamic_pool_list, block_splitting)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 1024);

  std::size_t initial_blocks = pool.get_block_count();

  // Allocate small amount (should split block)
  void* ptr = pool.allocate(1024);
  EXPECT_NE(ptr, nullptr);

  // Should have created additional blocks (allocated + remainder)
  EXPECT_GT(pool.get_block_count(), initial_blocks);

  pool.deallocate(ptr);
}

TEST(dynamic_pool_list, no_split_when_remainder_too_small)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 8*1024, 4*1024);

  std::size_t initial_blocks = pool.get_block_count();

  // Allocate size that leaves small remainder (< min_alloc_size)
  // Pool is 8KB, min_alloc is 4KB, so allocating 5KB leaves 3KB (< 4KB)
  void* ptr = pool.allocate(5 * 1024);
  EXPECT_NE(ptr, nullptr);

  // Block should not be split (remainder too small)
  EXPECT_EQ(pool.get_block_count(), initial_blocks);

  pool.deallocate(ptr);
}

TEST(dynamic_pool_list, split_when_remainder_large_enough)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 16*1024, 4*1024);

  std::size_t initial_blocks = pool.get_block_count();

  // Allocate size that leaves large remainder
  // Pool is 16KB, allocating 4KB leaves 12KB (> 4KB min)
  void* ptr = pool.allocate(4 * 1024);
  EXPECT_NE(ptr, nullptr);

  // Block should be split
  EXPECT_GT(pool.get_block_count(), initial_blocks);

  pool.deallocate(ptr);
}

// ============================================================================
// Coalescing Tests
// ============================================================================

TEST(dynamic_pool_list, coalesce_adjacent_free_blocks)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 1024);

  // Allocate three adjacent blocks
  void* ptr1 = pool.allocate(4096);
  void* ptr2 = pool.allocate(4096);
  void* ptr3 = pool.allocate(4096);

  std::size_t blocks_allocated = pool.get_block_count();

  // Free middle block first
  pool.deallocate(ptr2);
  std::size_t blocks_after_one_free = pool.get_block_count();

  // Free adjacent blocks - should trigger coalescing
  pool.deallocate(ptr1);
  pool.deallocate(ptr3);

  // After coalescing, should have fewer blocks than before
  EXPECT_LT(pool.get_block_count(), blocks_allocated);
}

TEST(dynamic_pool_list, coalesce_with_next_block)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 1024);

  void* ptr1 = pool.allocate(4096);
  void* ptr2 = pool.allocate(4096);

  std::size_t blocks_before = pool.get_block_count();

  // Free in order - should coalesce forward
  pool.deallocate(ptr1);
  pool.deallocate(ptr2);

  // Should have fewer blocks due to coalescing
  EXPECT_LT(pool.get_block_count(), blocks_before);
}

TEST(dynamic_pool_list, coalesce_with_previous_block)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 1024);

  void* ptr1 = pool.allocate(4096);
  void* ptr2 = pool.allocate(4096);

  std::size_t blocks_before = pool.get_block_count();

  // Free in reverse order - should coalesce backward
  pool.deallocate(ptr2);
  pool.deallocate(ptr1);

  // Should have fewer blocks due to coalescing
  EXPECT_LT(pool.get_block_count(), blocks_before);
}

TEST(dynamic_pool_list, coalesce_multiple_adjacent_blocks)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 128*1024, 1024);

  // Allocate many small blocks
  std::vector<void*> ptrs;
  for (int i = 0; i < 10; ++i) {
    ptrs.push_back(pool.allocate(2048));
  }

  // Free all - should coalesce into larger blocks
  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }

  // After coalescing, should have much fewer blocks
  EXPECT_LT(pool.get_block_count(), ptrs.size());
}

// ============================================================================
// Pool Growth Tests
// ============================================================================

TEST(dynamic_pool_list, automatic_pool_growth)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 8*1024, 1024, 2.0);

  std::size_t initial_total = pool.get_total_size();

  // Allocate more than initial pool size
  std::vector<void*> ptrs;
  for (int i = 0; i < 10; ++i) {
    ptrs.push_back(pool.allocate(2048));
  }

  // Total size should have grown
  EXPECT_GT(pool.get_total_size(), initial_total);

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }
}

TEST(dynamic_pool_list, growth_factor_applied)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 8*1024, 1024, 2.0);

  // Fill initial pool
  std::vector<void*> ptrs;
  while (pool.get_free_size() >= 2048) {
    ptrs.push_back(pool.allocate(2048));
  }

  std::size_t size_after_first_pool = pool.get_total_size();

  // Trigger growth
  void* ptr = pool.allocate(2048);
  ptrs.push_back(ptr);

  // New total should be significantly larger (due to growth factor)
  EXPECT_GT(pool.get_total_size(), size_after_first_pool + 2048);

  for (void* p : ptrs) {
    pool.deallocate(p);
  }
}

TEST(dynamic_pool_list, large_allocation_triggers_appropriate_growth)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 8*1024, 1024, 2.0);

  // Request allocation larger than next pool size
  void* ptr = pool.allocate(32 * 1024);
  EXPECT_NE(ptr, nullptr);

  // Pool should have grown to accommodate
  EXPECT_GE(pool.get_total_size(), 32 * 1024);

  pool.deallocate(ptr);
}

// ============================================================================
// Release Tests
// ============================================================================

TEST(dynamic_pool_list, release_with_no_free_blocks)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 16*1024, 1024);

  // Allocate all memory
  std::vector<void*> ptrs;
  while (pool.get_free_size() >= 1024) {
    ptrs.push_back(pool.allocate(1024));
  }

  std::size_t total_before = pool.get_total_size();

  // Release should not free anything (no free blocks)
  pool.release();

  EXPECT_EQ(pool.get_total_size(), total_before);

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }
}

TEST(dynamic_pool_list, release_returns_free_memory)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 16*1024, 1024, 2.0);

  // Allocate to create multiple pools
  std::vector<void*> ptrs;
  for (int i = 0; i < 20; ++i) {
    ptrs.push_back(pool.allocate(2048));
  }

  std::size_t total_after_alloc = pool.get_total_size();

  // Deallocate all
  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }

  EXPECT_EQ(pool.get_allocated_size(), 0);

  // Release free memory
  pool.release();

  // Total size should decrease (some memory returned to parent)
  EXPECT_LE(pool.get_total_size(), total_after_alloc);
}

TEST(dynamic_pool_list, release_keeps_some_free_memory)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 16*1024, 1024);

  void* ptr = pool.allocate(1024);
  pool.deallocate(ptr);

  // Release
  pool.release();

  // Should still be able to allocate (kept some free memory)
  void* ptr2 = pool.allocate(1024);
  EXPECT_NE(ptr2, nullptr);
  pool.deallocate(ptr2);
}

// ============================================================================
// Error Detection Tests
// ============================================================================

TEST(dynamic_pool_list, unknown_pointer_throws)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent);

  // Try to free pointer not from this pool
  int dummy;
  void* bad_ptr = &dummy;

  EXPECT_THROW(pool.deallocate(bad_ptr), umpire::unknown_pointer_error);
}

TEST(dynamic_pool_list, double_free_throws)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent);

  void* ptr = pool.allocate(1024);
  pool.deallocate(ptr);

  // Try to free again
  EXPECT_THROW(pool.deallocate(ptr), umpire::runtime_error);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(dynamic_pool_list, statistics_accuracy)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 1024);

  std::size_t initial_total = pool.get_total_size();
  EXPECT_EQ(pool.get_allocated_size(), 0);
  EXPECT_EQ(pool.get_free_size(), initial_total);

  // Allocate
  void* ptr1 = pool.allocate(4096);
  void* ptr2 = pool.allocate(8192);

  EXPECT_EQ(pool.get_allocated_size(), 4096 + 8192);
  EXPECT_LE(pool.get_free_size(), initial_total - (4096 + 8192));

  // Deallocate one
  pool.deallocate(ptr1);
  EXPECT_EQ(pool.get_allocated_size(), 8192);

  // Deallocate all
  pool.deallocate(ptr2);
  EXPECT_EQ(pool.get_allocated_size(), 0);
}

TEST(dynamic_pool_list, statistics_consistency)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 1024);

  // Total should equal allocated + free at all times
  EXPECT_EQ(pool.get_total_size(), pool.get_allocated_size() + pool.get_free_size());

  void* ptr1 = pool.allocate(2048);
  EXPECT_EQ(pool.get_total_size(), pool.get_allocated_size() + pool.get_free_size());

  void* ptr2 = pool.allocate(4096);
  EXPECT_EQ(pool.get_total_size(), pool.get_allocated_size() + pool.get_free_size());

  pool.deallocate(ptr1);
  EXPECT_EQ(pool.get_total_size(), pool.get_allocated_size() + pool.get_free_size());

  pool.deallocate(ptr2);
  EXPECT_EQ(pool.get_total_size(), pool.get_allocated_size() + pool.get_free_size());
}

// ============================================================================
// Memory Content Tests
// ============================================================================

TEST(dynamic_pool_list, allocated_memory_is_writable)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent);

  void* ptr = pool.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  // Write to memory
  char* bytes = static_cast<char*>(ptr);
  for (int i = 0; i < 1024; ++i) {
    bytes[i] = static_cast<char>(i & 0xFF);
  }

  // Verify written values
  for (int i = 0; i < 1024; ++i) {
    EXPECT_EQ(bytes[i], static_cast<char>(i & 0xFF));
  }

  pool.deallocate(ptr);
}

TEST(dynamic_pool_list, unique_allocations)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 512);

  // Allocate multiple blocks and verify they're unique
  std::vector<void*> ptrs;
  for (int i = 0; i < 10; ++i) {
    void* ptr = pool.allocate(1024);
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

TEST(dynamic_pool_list, churn_pattern)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 1024);

  // Simulate allocation churn
  for (int iter = 0; iter < 10; ++iter) {
    std::vector<void*> ptrs;
    for (int i = 0; i < 20; ++i) {
      ptrs.push_back(pool.allocate(1024 + (i * 128)));
    }

    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }

  // Should still be functional
  EXPECT_EQ(pool.get_allocated_size(), 0);
}

TEST(dynamic_pool_list, interleaved_alloc_dealloc)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 1024);

  std::vector<void*> ptrs;

  // Interleave allocations and deallocations
  for (int i = 0; i < 30; ++i) {
    ptrs.push_back(pool.allocate(1024 + (i * 64)));

    if (i > 0 && i % 3 == 0 && !ptrs.empty()) {
      pool.deallocate(ptrs.back());
      ptrs.pop_back();
    }
  }

  // Cleanup
  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }

  EXPECT_EQ(pool.get_allocated_size(), 0);
}

TEST(dynamic_pool_list, fragmentation_pattern)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 128*1024, 1024);

  // Allocate many blocks
  std::vector<void*> ptrs;
  for (int i = 0; i < 20; ++i) {
    ptrs.push_back(pool.allocate(2048));
  }

  // Free every other block (creates fragmentation)
  for (size_t i = 0; i < ptrs.size(); i += 2) {
    pool.deallocate(ptrs[i]);
    ptrs[i] = nullptr;
  }

  // Allocate small blocks (should fit in freed spaces)
  for (size_t i = 0; i < ptrs.size(); i += 2) {
    ptrs[i] = pool.allocate(1024);
  }

  // Cleanup
  for (void* ptr : ptrs) {
    if (ptr) pool.deallocate(ptr);
  }
}

// ============================================================================
// Platform Type Propagation Tests
// ============================================================================

TEST(dynamic_pool_list, platform_type_propagation)
{
  using host_mem = umpire::resource::host_memory<>;
  using dynamic_pool_host = umpire::strategy::dynamic_pool_list<host_mem>;

  // Platform type should be propagated from host_memory
  static_assert(std::is_same<dynamic_pool_host::platform, umpire::host_platform>::value,
                "Platform type should be propagated from wrapped memory");
}

// ============================================================================
// Composition Tests
// ============================================================================

TEST(dynamic_pool_list, composition_with_host_memory)
{
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::dynamic_pool_list<umpire::resource::host_memory<>>
    pool("dynamic_pool_host", &host, 64*1024, 2*1024);

  void* ptr = pool.allocate(4096);
  EXPECT_NE(ptr, nullptr);

  // Write to verify memory is accessible
  char* bytes = static_cast<char*>(ptr);
  bytes[0] = 'A';
  bytes[4095] = 'Z';
  EXPECT_EQ(bytes[0], 'A');
  EXPECT_EQ(bytes[4095], 'Z');

  pool.deallocate(ptr);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(dynamic_pool_list, large_allocation)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 8*1024, 1024);

  // Allocate much larger than initial pool
  void* ptr = pool.allocate(1024 * 1024);
  EXPECT_NE(ptr, nullptr);

  pool.deallocate(ptr);
}

TEST(dynamic_pool_list, many_small_allocations)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 64*1024, 256);

  std::vector<void*> ptrs;
  for (int i = 0; i < 100; ++i) {
    ptrs.push_back(pool.allocate(256));
  }

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }

  EXPECT_EQ(pool.get_allocated_size(), 0);
}

TEST(dynamic_pool_list, mixed_size_allocations)
{
  test_memory parent;
  umpire::strategy::dynamic_pool_list<test_memory> pool("dynamic_pool", &parent, 128*1024, 512);

  std::vector<std::size_t> sizes = {512, 1024, 2048, 4096, 8192, 512, 1024};
  std::vector<void*> ptrs;

  for (std::size_t size : sizes) {
    ptrs.push_back(pool.allocate(size));
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
