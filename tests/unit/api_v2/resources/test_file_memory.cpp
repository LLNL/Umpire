//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_FILE_RESOURCE)

#include "umpire/resource/file_memory.hpp"

#include "umpire/detail/registry.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <type_traits>

using namespace umpire::resource;

// Test that singleton returns the same instance each time
TEST(file_memory, singleton_returns_same_instance)
{
  auto& inst1 = file_memory<>::get();
  auto& inst2 = file_memory<>::get();

  EXPECT_EQ(&inst1, &inst2);
  EXPECT_EQ(inst1.get_name(), "FILE");
}

// Test that custom instance can be constructed with custom name
TEST(file_memory, custom_instance_with_custom_name)
{
  file_memory<> custom("CUSTOM_FILE");
  EXPECT_EQ(custom.get_name(), "CUSTOM_FILE");
}

// Test basic allocation: allocate, write, read back, deallocate
TEST(file_memory, allocate_write_read_deallocate_roundtrip)
{
  file_memory<> mem("TEST_FILE");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  char* bytes = static_cast<char*>(ptr);
  bytes[0] = 'A';
  bytes[1023] = 'Z';
  EXPECT_EQ(bytes[0], 'A');
  EXPECT_EQ(bytes[1023], 'Z');

  mem.deallocate(ptr);
}

// Test tracking enabled: allocate, verify registry statistics, deallocate, verify removed
TEST(file_memory, tracking_enabled_records_allocation)
{
  file_memory<true> mem("TRACKED_FILE");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 1024u);
  EXPECT_EQ(mem.get_highwatermark(), 1024u);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0u);
  EXPECT_EQ(mem.get_highwatermark(), 1024u);
}

// Test tracking disabled: verify no registry interaction
TEST(file_memory, tracking_disabled_no_registry_interaction)
{
  file_memory<false> mem("UNTRACKED_FILE");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 0u);
  EXPECT_EQ(mem.get_highwatermark(), 0u);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0u);
  EXPECT_EQ(mem.get_highwatermark(), 0u);
}

// Test zero-size allocation returns nullptr
TEST(file_memory, zero_size_allocation_returns_nullptr)
{
  file_memory<> mem("ZERO_FILE");

  void* ptr = mem.allocate(0);
  EXPECT_EQ(ptr, nullptr);

  // Deallocate should be safe no-op
  mem.deallocate(ptr);
}

// Test nullptr deallocation is safe no-op
TEST(file_memory, nullptr_deallocation_is_safe)
{
  file_memory<> mem("NULL_FILE");

  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

// Test that deallocating an unknown pointer is a safe no-op
TEST(file_memory, unknown_pointer_deallocation_is_safe)
{
  file_memory<> mem("UNKNOWN_FILE");

  int stack_value = 42;
  EXPECT_NO_THROW(mem.deallocate(&stack_value));
}

// Test allocation is visible in the shared v2 registry while live
TEST(file_memory, registry_visibility)
{
  file_memory<> mem("REGISTRY_FILE");

  void* ptr = mem.allocate(256);
  ASSERT_NE(ptr, nullptr);

  auto found = umpire::detail::registry::get().find_allocation(ptr);
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(found->ptr, ptr);
  EXPECT_EQ(found->size, 256u);

  mem.deallocate(ptr);

  EXPECT_FALSE(umpire::detail::registry::get().find_allocation(ptr).has_value());
}

// Test multiple concurrent (live at the same time) allocations track correctly
TEST(file_memory, multiple_allocations_correct_statistics)
{
  file_memory<> mem("MULTI_FILE");

  void* a = mem.allocate(100);
  EXPECT_EQ(mem.get_current_size(), 100u);
  EXPECT_EQ(mem.get_highwatermark(), 100u);

  void* b = mem.allocate(200);
  EXPECT_EQ(mem.get_current_size(), 300u);
  EXPECT_EQ(mem.get_highwatermark(), 300u);

  void* c = mem.allocate(150);
  EXPECT_EQ(mem.get_current_size(), 450u);
  EXPECT_EQ(mem.get_highwatermark(), 450u);

  // Each allocation should be backed by an independent, writable mapping.
  static_cast<char*>(a)[0] = 'a';
  static_cast<char*>(b)[0] = 'b';
  static_cast<char*>(c)[0] = 'c';
  EXPECT_EQ(static_cast<char*>(a)[0], 'a');
  EXPECT_EQ(static_cast<char*>(b)[0], 'b');
  EXPECT_EQ(static_cast<char*>(c)[0], 'c');

  mem.deallocate(b);
  EXPECT_EQ(mem.get_current_size(), 250u);
  EXPECT_EQ(mem.get_highwatermark(), 450u);

  mem.deallocate(a);
  EXPECT_EQ(mem.get_current_size(), 150u);
  EXPECT_EQ(mem.get_highwatermark(), 450u);

  mem.deallocate(c);
  EXPECT_EQ(mem.get_current_size(), 0u);
  EXPECT_EQ(mem.get_highwatermark(), 450u);
}

// Test platform type is correct
TEST(file_memory, platform_type_is_host)
{
  file_memory<> mem("PLATFORM_FILE");
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::host);
}

// Test convenience aliases
TEST(file_memory, convenience_aliases)
{
  static_assert(default_file_memory::tracking_enabled == true,
                "default_file_memory should have tracking enabled");
  static_assert(fast_file_memory::tracking_enabled == false,
                "fast_file_memory should have tracking disabled");
}

// Test that allocations from different instances are independent
TEST(file_memory, independent_instances)
{
  file_memory<> mem1("FILE1");
  file_memory<> mem2("FILE2");

  void* ptr1 = mem1.allocate(100);
  void* ptr2 = mem2.allocate(200);
  ASSERT_NE(ptr1, nullptr);
  ASSERT_NE(ptr2, nullptr);

  EXPECT_EQ(mem1.get_current_size(), 100u);
  EXPECT_EQ(mem2.get_current_size(), 200u);

  mem1.deallocate(ptr1);
  mem2.deallocate(ptr2);
}

// Test many small allocations succeed and are independently addressable
TEST(file_memory, many_small_allocations)
{
  file_memory<> mem("MANY_SMALL_FILE");

  const int num_allocs = 16;
  void* ptrs[num_allocs];

  for (int i = 0; i < num_allocs; ++i) {
    ptrs[i] = mem.allocate(64);
    ASSERT_NE(ptrs[i], nullptr);
    static_cast<char*>(ptrs[i])[0] = static_cast<char>('A' + i);
  }

  EXPECT_EQ(mem.get_current_size(), static_cast<std::size_t>(num_allocs) * 64u);

  for (int i = 0; i < num_allocs; ++i) {
    EXPECT_EQ(static_cast<char*>(ptrs[i])[0], static_cast<char>('A' + i));
  }

  for (int i = 0; i < num_allocs; ++i) {
    mem.deallocate(ptrs[i]);
  }

  EXPECT_EQ(mem.get_current_size(), 0u);
}

// Test destroying a resource with outstanding allocations cleans up backing files
TEST(file_memory, destructor_releases_outstanding_allocations)
{
  void* ptr = nullptr;
  {
    file_memory<> mem("LEAKED_FILE");
    ptr = mem.allocate(128);
    ASSERT_NE(ptr, nullptr);
    // Resource destructor runs here without an explicit deallocate() call.
  }

  // The pointer should no longer be tracked by the shared registry once the
  // owning resource has torn down its outstanding allocations.
  EXPECT_FALSE(umpire::detail::registry::get().find_allocation(ptr).has_value());
}

// Test type traits
TEST(file_memory, type_traits)
{
  using tracked_file = file_memory<true>;
  using untracked_file = file_memory<false>;

  static_assert(std::is_same_v<tracked_file::platform, umpire::host_platform>,
                "Platform should be host_platform");

  static_assert(tracked_file::tracking_enabled == true,
                "tracking_enabled should be true");
  static_assert(untracked_file::tracking_enabled == false,
                "tracking_enabled should be false");
}

#endif // UMPIRE_ENABLE_FILE_RESOURCE
