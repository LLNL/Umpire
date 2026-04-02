//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/detail/registry.hpp"
#include "umpire/memory_resource.hpp"
#include "umpire/op/reallocate.hpp"

#include "camp/resource/host.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <algorithm>
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

camp::resources::Resource host_resource()
{
  return camp::resources::Resource{camp::resources::Host{}};
}

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
  auto record = umpire::detail::registry::get().find_allocation(ptr);

  // Verify allocation is tracked
  ASSERT_TRUE(record.has_value());
  EXPECT_EQ(record->strategy, &mem);
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

TEST(memory_resource, tracked_owner_reallocate_preserves_typed_contents)
{
  test_resource_tracked mem;

  auto* values = static_cast<int*>(mem.allocate(4 * sizeof(int)));
  for (int i = 0; i < 4; ++i) {
    values[i] = i + 17;
  }

  values = umpire::reallocate(&values, 8);

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(values[i], i + 17);
  }
  EXPECT_EQ(mem.get_current_size(), 8 * sizeof(int));

  mem.deallocate(values);
}

TEST(memory_resource, tracked_owner_async_reallocate_preserves_byte_contents)
{
  test_resource_tracked mem;
  auto resource = host_resource();

  void* ptr = mem.allocate(8);
  auto* bytes = static_cast<unsigned char*>(ptr);
  std::fill(bytes, bytes + 8, static_cast<unsigned char>(0xA5));

  camp::resources::Event event = umpire::reallocate(&ptr, 16, resource);
  event.wait();

  bytes = static_cast<unsigned char*>(ptr);
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(bytes[i], 0xA5);
  }
  EXPECT_EQ(mem.get_current_size(), 16);

  mem.deallocate(ptr);
}
