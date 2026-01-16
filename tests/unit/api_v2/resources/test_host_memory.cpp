//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/host_memory.hpp"

#include <gtest/gtest.h>
#include <type_traits>

using namespace umpire::resource;

// Test that singleton returns the same instance each time
TEST(host_memory, singleton_returns_same_instance)
{
  auto& inst1 = host_memory<>::get();
  auto& inst2 = host_memory<>::get();

  // Should be the exact same instance
  EXPECT_EQ(&inst1, &inst2);
  EXPECT_EQ(inst1.get_name(), "HOST");
}

// Test that custom instance can be constructed with custom name
TEST(host_memory, custom_instance_with_custom_name)
{
  host_memory<> custom("CUSTOM_HOST");
  EXPECT_EQ(custom.get_name(), "CUSTOM_HOST");
}

// Test basic allocation: allocate(1024), verify non-null, deallocate
TEST(host_memory, basic_allocation)
{
  host_memory<> mem("TEST_HOST");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  // Verify we can write to the memory
  char* bytes = static_cast<char*>(ptr);
  bytes[0] = 'A';
  bytes[1023] = 'Z';
  EXPECT_EQ(bytes[0], 'A');
  EXPECT_EQ(bytes[1023], 'Z');

  mem.deallocate(ptr);
}

// Test tracking enabled: allocate, verify registry has record, deallocate, verify removed
TEST(host_memory, tracking_enabled_records_allocation)
{
  host_memory<malloc_allocator, true> mem("TRACKED_HOST");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  // Verify allocation is tracked
  EXPECT_EQ(mem.get_current_size(), 1024);
  EXPECT_EQ(mem.get_highwatermark(), 1024);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 1024);
}

// Test tracking disabled: verify no registry interaction
TEST(host_memory, tracking_disabled_no_registry_interaction)
{
  host_memory<malloc_allocator, false> mem("UNTRACKED_HOST");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  // With tracking disabled, statistics should remain at 0
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 0);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 0);
}

// Test large allocation
TEST(host_memory, large_allocation)
{
  host_memory<> mem("LARGE_HOST");

  // Allocate 100MB (should succeed on most systems)
  std::size_t size = 100 * 1024 * 1024;
  void* ptr = mem.allocate(size);
  ASSERT_NE(ptr, nullptr);

  // Write to first and last bytes to verify it's accessible
  char* bytes = static_cast<char*>(ptr);
  bytes[0] = 'A';
  bytes[size - 1] = 'Z';
  EXPECT_EQ(bytes[0], 'A');
  EXPECT_EQ(bytes[size - 1], 'Z');

  mem.deallocate(ptr);
}

// Test zero-size allocation returns nullptr
TEST(host_memory, zero_size_allocation_returns_nullptr)
{
  host_memory<> mem("ZERO_HOST");

  void* ptr = mem.allocate(0);
  EXPECT_EQ(ptr, nullptr);

  // Deallocate should be safe no-op
  mem.deallocate(ptr);
}

// Test nullptr deallocation is safe no-op
TEST(host_memory, nullptr_deallocation_is_safe)
{
  host_memory<> mem("NULL_HOST");

  // Should not crash or throw
  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

// Test multiple allocations with correct statistics
TEST(host_memory, multiple_allocations_correct_statistics)
{
  host_memory<> mem("MULTI_HOST");

  void* a = mem.allocate(100);
  EXPECT_EQ(mem.get_current_size(), 100);
  EXPECT_EQ(mem.get_highwatermark(), 100);

  void* b = mem.allocate(200);
  EXPECT_EQ(mem.get_current_size(), 300);
  EXPECT_EQ(mem.get_highwatermark(), 300);

  void* c = mem.allocate(150);
  EXPECT_EQ(mem.get_current_size(), 450);
  EXPECT_EQ(mem.get_highwatermark(), 450);

  mem.deallocate(b);
  EXPECT_EQ(mem.get_current_size(), 250);
  EXPECT_EQ(mem.get_highwatermark(), 450);

  mem.deallocate(a);
  EXPECT_EQ(mem.get_current_size(), 150);
  EXPECT_EQ(mem.get_highwatermark(), 450);

  mem.deallocate(c);
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 450);
}

// Test exception handling: force allocation failure
TEST(host_memory, allocation_failure_throws_exception)
{
  host_memory<> mem("FAIL_HOST");

  // Try to allocate an impossibly large amount (close to SIZE_MAX)
  // This should fail and throw out_of_memory_error
  std::size_t huge_size = std::numeric_limits<std::size_t>::max() - 1024;
  EXPECT_THROW(mem.allocate(huge_size), umpire::out_of_memory_error);
}

// Test platform type is correct
TEST(host_memory, platform_type_is_host)
{
  host_memory<> mem("PLATFORM_HOST");
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::host);
}

// Test convenience aliases
TEST(host_memory, convenience_aliases)
{
  // default_host_memory should have tracking enabled
  static_assert(default_host_memory::tracking_enabled == true,
                "default_host_memory should have tracking enabled");

  // fast_host_memory should have tracking disabled
  static_assert(fast_host_memory::tracking_enabled == false,
                "fast_host_memory should have tracking disabled");
}

// Test type traits
TEST(host_memory, type_traits)
{
  using tracked_host = host_memory<malloc_allocator, true>;
  using untracked_host = host_memory<malloc_allocator, false>;

  // Platform type
  static_assert(std::is_same_v<tracked_host::platform, umpire::host_platform>,
                "Platform should be host_platform");

  // Allocator type
  static_assert(std::is_same_v<tracked_host::allocator_type, malloc_allocator>,
                "Allocator type should be malloc_allocator");

  // Tracking flag
  static_assert(tracked_host::tracking_enabled == true,
                "tracking_enabled should be true");
  static_assert(untracked_host::tracking_enabled == false,
                "tracking_enabled should be false");
}

// Test that allocations from different instances are independent
TEST(host_memory, independent_instances)
{
  host_memory<> mem1("HOST1");
  host_memory<> mem2("HOST2");

  void* ptr1 = mem1.allocate(100);
  void* ptr2 = mem2.allocate(200);

  EXPECT_EQ(mem1.get_current_size(), 100);
  EXPECT_EQ(mem2.get_current_size(), 200);

  mem1.deallocate(ptr1);
  mem2.deallocate(ptr2);
}

// Test many small allocations
TEST(host_memory, many_small_allocations)
{
  host_memory<> mem("MANY_SMALL");

  const int num_allocs = 1000;
  void* ptrs[num_allocs];

  // Allocate many small blocks
  for (int i = 0; i < num_allocs; ++i) {
    ptrs[i] = mem.allocate(16);
    ASSERT_NE(ptrs[i], nullptr);
  }

  EXPECT_EQ(mem.get_current_size(), num_allocs * 16);

  // Deallocate all
  for (int i = 0; i < num_allocs; ++i) {
    mem.deallocate(ptrs[i]);
  }

  EXPECT_EQ(mem.get_current_size(), 0);
}
