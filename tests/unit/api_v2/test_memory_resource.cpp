//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/detail/registry.hpp"
#include "umpire/memory_resource.hpp"

#include <cstdlib>
#include <gtest/gtest.h>
#include <type_traits>

namespace {

// Test resource with tracking enabled (default)
class test_resource_tracked : public umpire::memory_resource<umpire::host_platform, std::allocator<char>, true> {
public:
  test_resource_tracked() : memory_resource("test_tracked") {}

  void* allocate(std::size_t size) override {
    return allocate_impl(size);
  }

  void deallocate(void* ptr) override {
    deallocate_impl(ptr, 0);
  }
};

// Test resource with tracking disabled
class test_resource_untracked : public umpire::memory_resource<umpire::host_platform, std::allocator<char>, false> {
public:
  test_resource_untracked() : memory_resource("test_untracked") {}

  void* allocate(std::size_t size) override {
    return allocate_impl(size);
  }

  void deallocate(void* ptr) override {
    deallocate_impl(ptr, 0);
  }
};

} // namespace

// Test platform type propagation
TEST(memory_resource, platform_type_propagation)
{
  // Verify the platform type alias is correctly set
  static_assert(std::is_same_v<test_resource_tracked::platform, umpire::host_platform>,
                "Platform type alias should be host_platform");

  static_assert(std::is_same_v<test_resource_untracked::platform, umpire::host_platform>,
                "Platform type alias should be host_platform");
}

// Test get_platform() returns correct value
TEST(memory_resource, get_platform_returns_correct_value)
{
  test_resource_tracked mem;
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::host);
}

// Test tracking enabled: allocate and verify registry has record
TEST(memory_resource, tracking_enabled_records_allocation)
{
  test_resource_tracked mem;

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  // Verify allocation is tracked
  EXPECT_EQ(mem.get_current_size(), 1024);
  EXPECT_EQ(mem.get_highwatermark(), 1024);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 1024);
}

// Test tracking disabled: allocate and verify no registry record
TEST(memory_resource, tracking_disabled_no_registry_record)
{
  test_resource_untracked mem;

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  // With tracking disabled, statistics should remain at 0
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 0);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 0);
}

// Test allocator type alias
TEST(memory_resource, allocator_type_alias)
{
  static_assert(std::is_same_v<test_resource_tracked::allocator_type, std::allocator<char>>,
                "Allocator type alias should be std::allocator<char>");
}

// Test tracking_enabled static member
TEST(memory_resource, tracking_enabled_constexpr)
{
  static_assert(test_resource_tracked::tracking_enabled == true,
                "tracking_enabled should be true for tracked resource");

  static_assert(test_resource_untracked::tracking_enabled == false,
                "tracking_enabled should be false for untracked resource");
}

// Test multiple allocations with tracking
TEST(memory_resource, multiple_allocations_tracked)
{
  test_resource_tracked mem;

  void* a = mem.allocate(100);
  EXPECT_EQ(mem.get_current_size(), 100);

  void* b = mem.allocate(200);
  EXPECT_EQ(mem.get_current_size(), 300);
  EXPECT_EQ(mem.get_highwatermark(), 300);

  mem.deallocate(a);
  EXPECT_EQ(mem.get_current_size(), 200);
  EXPECT_EQ(mem.get_highwatermark(), 300);

  mem.deallocate(b);
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 300);
}

// Test that default allocator is used correctly
TEST(memory_resource, default_allocator_usage)
{
  // This tests that default_allocator_for provides the correct type
  using default_alloc = typename umpire::default_allocator_for<umpire::host_platform>::type;
  static_assert(std::is_same_v<default_alloc, std::allocator<char>>,
                "Default allocator for host_platform should be std::allocator<char>");
}
