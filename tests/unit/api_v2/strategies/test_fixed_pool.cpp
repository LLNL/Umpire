//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/fixed_pool.hpp"
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
  // Match the platform alias expected by strategy::fixed_pool and other templates
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

TEST(fixed_pool, construct_with_valid_parameters)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64);

  EXPECT_EQ(pool.get_parent(), &parent);
  EXPECT_EQ(pool.get_name(), "fixed_pool");
  EXPECT_EQ(pool.get_object_size(), 64);
  EXPECT_EQ(pool.get_objects_per_pool(), 1024);  // Default
  EXPECT_GT(pool.get_total_objects(), 0);  // Pre-allocated pool
  EXPECT_EQ(pool.get_free_objects(), pool.get_total_objects());
  EXPECT_EQ(pool.get_allocated_objects(), 0);
  EXPECT_EQ(pool.get_pool_count(), 1);
}

TEST(fixed_pool, construct_with_custom_objects_per_pool)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 128, 512);

  EXPECT_EQ(pool.get_object_size(), 128);
  EXPECT_EQ(pool.get_objects_per_pool(), 512);
  EXPECT_EQ(pool.get_total_objects(), 512);  // One pool with 512 objects
  EXPECT_EQ(pool.get_free_objects(), 512);
}

TEST(fixed_pool, construct_with_nullptr_throws)
{
  EXPECT_THROW(
    umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", nullptr, 64),
    std::invalid_argument
  );
}

TEST(fixed_pool, construct_with_zero_object_size_throws)
{
  test_memory parent;
  EXPECT_THROW(
    umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 0),
    std::invalid_argument
  );
}

TEST(fixed_pool, construct_with_zero_objects_per_pool_throws)
{
  test_memory parent;
  EXPECT_THROW(
    umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 0),
    std::invalid_argument
  );
}

TEST(fixed_pool, get_platform_delegates_to_parent)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64);

  EXPECT_EQ(pool.get_platform(), parent.get_platform());
  EXPECT_EQ(pool.get_platform(), umpire::resource::Platform::host);
}

// ============================================================================
// Basic Allocation Tests
// ============================================================================

TEST(fixed_pool, basic_allocation)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64);

  void* ptr = pool.allocate(64);
  EXPECT_NE(ptr, nullptr);

  // Verify statistics updated
  EXPECT_EQ(pool.get_allocated_objects(), 1);
  EXPECT_EQ(pool.get_free_objects(), pool.get_total_objects() - 1);

  pool.deallocate(ptr);
  EXPECT_EQ(pool.get_allocated_objects(), 0);
  EXPECT_EQ(pool.get_free_objects(), pool.get_total_objects());
}

TEST(fixed_pool, multiple_allocations)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 10);

  std::vector<void*> ptrs;
  for (int i = 0; i < 5; ++i) {
    void* ptr = pool.allocate(64);
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  EXPECT_EQ(pool.get_allocated_objects(), 5);
  EXPECT_EQ(pool.get_free_objects(), 5);

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }

  EXPECT_EQ(pool.get_allocated_objects(), 0);
  EXPECT_EQ(pool.get_free_objects(), 10);
}

TEST(fixed_pool, allocation_with_wrong_size_throws)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64);

  // Allocating with wrong size should throw
  EXPECT_THROW(pool.allocate(32), std::invalid_argument);
  EXPECT_THROW(pool.allocate(128), std::invalid_argument);
  EXPECT_THROW(pool.allocate(0), std::invalid_argument);
}

TEST(fixed_pool, nullptr_deallocation_is_safe)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64);

  // Should not crash or change statistics
  std::size_t free_before = pool.get_free_objects();
  EXPECT_NO_THROW(pool.deallocate(nullptr));
  EXPECT_EQ(pool.get_free_objects(), free_before);
}

// ============================================================================
// Pool Growth Tests
// ============================================================================

TEST(fixed_pool, automatic_pool_growth)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 10);

  // Allocate more objects than initial pool size
  std::vector<void*> ptrs;
  for (int i = 0; i < 15; ++i) {
    void* ptr = pool.allocate(64);
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Should have allocated a second pool
  EXPECT_EQ(pool.get_pool_count(), 2);
  EXPECT_EQ(pool.get_total_objects(), 20);
  EXPECT_EQ(pool.get_allocated_objects(), 15);
  EXPECT_EQ(pool.get_free_objects(), 5);

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }
}

TEST(fixed_pool, multiple_pool_allocations)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 32, 5);

  // Allocate enough to trigger multiple pool allocations
  std::vector<void*> ptrs;
  for (int i = 0; i < 23; ++i) {
    void* ptr = pool.allocate(32);
    EXPECT_NE(ptr, nullptr);
    ptrs.push_back(ptr);
  }

  // Should have 5 pools (5, 10, 15, 20, 25 objects)
  EXPECT_EQ(pool.get_pool_count(), 5);
  EXPECT_EQ(pool.get_total_objects(), 25);
  EXPECT_EQ(pool.get_allocated_objects(), 23);

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }

  EXPECT_EQ(pool.get_allocated_objects(), 0);
  EXPECT_EQ(pool.get_free_objects(), 25);
}

// ============================================================================
// Release Tests
// ============================================================================

TEST(fixed_pool, release_with_no_free_pools)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 10);

  // Allocate some objects
  void* ptr1 = pool.allocate(64);
  void* ptr2 = pool.allocate(64);

  std::size_t pools_before = pool.get_pool_count();

  // Release should not remove any pools (not enough free objects)
  pool.release();

  EXPECT_EQ(pool.get_pool_count(), pools_before);

  pool.deallocate(ptr1);
  pool.deallocate(ptr2);
}

TEST(fixed_pool, release_with_multiple_pools)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 10);

  // Allocate objects to create multiple pools
  std::vector<void*> ptrs;
  for (int i = 0; i < 25; ++i) {
    ptrs.push_back(pool.allocate(64));
  }

  EXPECT_EQ(pool.get_pool_count(), 3);

  // Deallocate all
  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }

  EXPECT_EQ(pool.get_free_objects(), 30);

  // Release should remove excess pools but keep at least one
  pool.release();

  EXPECT_GE(pool.get_pool_count(), 1);
  EXPECT_LE(pool.get_pool_count(), 3);
}

TEST(fixed_pool, release_keeps_at_least_one_pool)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 10);

  // All objects are free in the initial pool
  EXPECT_EQ(pool.get_pool_count(), 1);
  EXPECT_EQ(pool.get_free_objects(), 10);

  // Release should keep at least one pool
  pool.release();

  EXPECT_EQ(pool.get_pool_count(), 1);
  EXPECT_GT(pool.get_free_objects(), 0);

  // Should still be able to allocate
  void* ptr = pool.allocate(64);
  EXPECT_NE(ptr, nullptr);
  pool.deallocate(ptr);
}

TEST(fixed_pool, release_only_removes_fully_free_pool)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 10);

  // Two pools: ptrs[0..9] come from the first pool, ptrs[10..19] from the
  // second (the free list is fully drained before each growth).
  std::vector<void*> ptrs;
  for (int i = 0; i < 20; ++i) {
    ptrs.push_back(pool.allocate(64));
  }
  EXPECT_EQ(pool.get_pool_count(), 2);

  // Free the FIRST pool's objects; the second pool stays fully live.
  for (int i = 0; i < 10; ++i) {
    pool.deallocate(ptrs[i]);
  }
  EXPECT_EQ(pool.get_free_objects(), 10);

  pool.release();

  // Only the fully-free first pool may be released; live allocations from
  // the second pool must survive.
  EXPECT_EQ(pool.get_pool_count(), 1);
  EXPECT_EQ(pool.get_allocated_objects(), 10);
  EXPECT_EQ(pool.get_free_objects(), 0);

  // Surviving pointers remain usable.
  for (int i = 10; i < 20; ++i) {
    std::memset(ptrs[i], 0xAB, 64);
    EXPECT_EQ(static_cast<unsigned char*>(ptrs[i])[0], 0xAB);
    EXPECT_EQ(static_cast<unsigned char*>(ptrs[i])[63], 0xAB);
  }

  // New allocations must not alias live pointers.
  std::vector<void*> fresh;
  for (int i = 0; i < 10; ++i) {
    void* p = pool.allocate(64);
    EXPECT_NE(p, nullptr);
    for (int j = 10; j < 20; ++j) {
      EXPECT_NE(p, ptrs[j]);
    }
    fresh.push_back(p);
  }

  for (void* p : fresh) {
    pool.deallocate(p);
  }
  for (int i = 10; i < 20; ++i) {
    pool.deallocate(ptrs[i]);
  }
}

TEST(fixed_pool, release_removes_fully_free_middle_pool)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 10);

  // Three pools: indices 0-9, 10-19, 20-29 map to pools 1, 2, 3.
  std::vector<void*> ptrs;
  for (int i = 0; i < 30; ++i) {
    ptrs.push_back(pool.allocate(64));
  }
  EXPECT_EQ(pool.get_pool_count(), 3);

  // Free only the middle pool's objects.
  for (int i = 10; i < 20; ++i) {
    pool.deallocate(ptrs[i]);
  }

  pool.release();

  EXPECT_EQ(pool.get_pool_count(), 2);
  EXPECT_EQ(pool.get_allocated_objects(), 20);
  EXPECT_EQ(pool.get_free_objects(), 0);

  // Live pointers from the first and third pools remain usable.
  for (int i = 0; i < 10; ++i) {
    std::memset(ptrs[i], 0x5C, 64);
    EXPECT_EQ(static_cast<unsigned char*>(ptrs[i])[63], 0x5C);
  }
  for (int i = 20; i < 30; ++i) {
    std::memset(ptrs[i], 0x5C, 64);
    EXPECT_EQ(static_cast<unsigned char*>(ptrs[i])[63], 0x5C);
  }

  for (int i = 0; i < 10; ++i) {
    pool.deallocate(ptrs[i]);
  }
  for (int i = 20; i < 30; ++i) {
    pool.deallocate(ptrs[i]);
  }
}

TEST(fixed_pool, release_with_no_fully_free_pool_is_noop)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 10);

  std::vector<void*> ptrs;
  for (int i = 0; i < 20; ++i) {
    ptrs.push_back(pool.allocate(64));
  }
  EXPECT_EQ(pool.get_pool_count(), 2);

  // Free every other object: 10 free objects total, but 5 in each pool, so
  // neither pool is fully free and nothing may be released.
  for (int i = 0; i < 20; i += 2) {
    pool.deallocate(ptrs[i]);
  }
  EXPECT_EQ(pool.get_free_objects(), 10);

  pool.release();

  EXPECT_EQ(pool.get_pool_count(), 2);
  EXPECT_EQ(pool.get_free_objects(), 10);
  EXPECT_EQ(pool.get_allocated_objects(), 10);

  // All still-allocated pointers remain usable.
  for (int i = 1; i < 20; i += 2) {
    std::memset(ptrs[i], 0x7E, 64);
    EXPECT_EQ(static_cast<unsigned char*>(ptrs[i])[0], 0x7E);
  }

  // The freed slots can be reallocated.
  std::vector<void*> fresh;
  for (int i = 0; i < 10; ++i) {
    fresh.push_back(pool.allocate(64));
  }
  EXPECT_EQ(pool.get_free_objects(), 0);

  for (void* p : fresh) {
    pool.deallocate(p);
  }
  for (int i = 1; i < 20; i += 2) {
    pool.deallocate(ptrs[i]);
  }
}

// ============================================================================
// Allocation/Deallocation Pattern Tests
// ============================================================================

TEST(fixed_pool, churn_pattern)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 100);

  // Simulate object churn: allocate and deallocate repeatedly
  for (int iter = 0; iter < 10; ++iter) {
    std::vector<void*> ptrs;
    for (int i = 0; i < 50; ++i) {
      ptrs.push_back(pool.allocate(64));
    }

    for (void* ptr : ptrs) {
      pool.deallocate(ptr);
    }
  }

  // Should still have just one pool (no growth needed)
  EXPECT_EQ(pool.get_pool_count(), 1);
  EXPECT_EQ(pool.get_allocated_objects(), 0);
}

TEST(fixed_pool, interleaved_alloc_dealloc)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 128, 20);

  std::vector<void*> ptrs;

  // Interleave allocations and deallocations
  for (int i = 0; i < 30; ++i) {
    ptrs.push_back(pool.allocate(128));

    if (i > 0 && i % 3 == 0 && !ptrs.empty()) {
      pool.deallocate(ptrs.back());
      ptrs.pop_back();
    }
  }

  // Cleanup
  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }

  EXPECT_EQ(pool.get_allocated_objects(), 0);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(fixed_pool, statistics_accuracy)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 10);

  // Initial state
  EXPECT_EQ(pool.get_total_objects(), 10);
  EXPECT_EQ(pool.get_free_objects(), 10);
  EXPECT_EQ(pool.get_allocated_objects(), 0);

  // Allocate 3 objects
  void* ptr1 = pool.allocate(64);
  void* ptr2 = pool.allocate(64);
  void* ptr3 = pool.allocate(64);

  EXPECT_EQ(pool.get_total_objects(), 10);
  EXPECT_EQ(pool.get_free_objects(), 7);
  EXPECT_EQ(pool.get_allocated_objects(), 3);

  // Deallocate 1
  pool.deallocate(ptr2);

  EXPECT_EQ(pool.get_total_objects(), 10);
  EXPECT_EQ(pool.get_free_objects(), 8);
  EXPECT_EQ(pool.get_allocated_objects(), 2);

  // Deallocate remaining
  pool.deallocate(ptr1);
  pool.deallocate(ptr3);

  EXPECT_EQ(pool.get_total_objects(), 10);
  EXPECT_EQ(pool.get_free_objects(), 10);
  EXPECT_EQ(pool.get_allocated_objects(), 0);
}

TEST(fixed_pool, statistics_with_growth)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 5);

  EXPECT_EQ(pool.get_total_objects(), 5);

  // Allocate to trigger growth
  std::vector<void*> ptrs;
  for (int i = 0; i < 8; ++i) {
    ptrs.push_back(pool.allocate(64));
  }

  EXPECT_EQ(pool.get_pool_count(), 2);
  EXPECT_EQ(pool.get_total_objects(), 10);
  EXPECT_EQ(pool.get_allocated_objects(), 8);
  EXPECT_EQ(pool.get_free_objects(), 2);

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }
}

// ============================================================================
// Memory Content Tests
// ============================================================================

TEST(fixed_pool, allocated_memory_is_writable)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 256);

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

TEST(fixed_pool, unique_allocations)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 10);

  // Allocate multiple objects and verify they're unique
  std::vector<void*> ptrs;
  for (int i = 0; i < 10; ++i) {
    void* ptr = pool.allocate(64);
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
// Platform Type Propagation Tests
// ============================================================================

TEST(fixed_pool, platform_type_propagation)
{
  using host_mem = umpire::resource::host_memory<>;
  using fixed_pool_host = umpire::strategy::fixed_pool<host_mem>;

  // Platform type should be propagated from host_memory
  static_assert(std::is_same<fixed_pool_host::platform, umpire::host_platform>::value,
                "Platform type should be propagated from wrapped memory");
}

// ============================================================================
// Composition Tests
// ============================================================================

TEST(fixed_pool, composition_with_host_memory)
{
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::fixed_pool<umpire::resource::host_memory<>>
    pool("fixed_pool_host", &host, 1024);

  void* ptr = pool.allocate(1024);
  EXPECT_NE(ptr, nullptr);

  // Write to verify memory is accessible
  char* bytes = static_cast<char*>(ptr);
  bytes[0] = 'A';
  bytes[1023] = 'Z';
  EXPECT_EQ(bytes[0], 'A');
  EXPECT_EQ(bytes[1023], 'Z');

  pool.deallocate(ptr);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(fixed_pool, large_object_size)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 1024 * 1024);

  void* ptr = pool.allocate(1024 * 1024);
  EXPECT_NE(ptr, nullptr);

  pool.deallocate(ptr);
}

TEST(fixed_pool, small_objects_per_pool)
{
  test_memory parent;
  umpire::strategy::fixed_pool<test_memory> pool("fixed_pool", &parent, 64, 1);

  EXPECT_EQ(pool.get_objects_per_pool(), 1);
  EXPECT_EQ(pool.get_total_objects(), 1);

  // Allocate to trigger multiple pools
  void* ptr1 = pool.allocate(64);
  void* ptr2 = pool.allocate(64);

  EXPECT_EQ(pool.get_pool_count(), 2);

  pool.deallocate(ptr1);
  pool.deallocate(ptr2);
}
