//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/shared_memory.hpp"

#include "umpire/detail/registry.hpp"

#include <gtest/gtest.h>
#include <type_traits>

using namespace umpire::resource;

namespace {
// Keep segment names short: macOS enforces a tight limit (historically
// PSHMNAMLEN == 31 bytes, including the leading '/') on shm_open() names.
constexpr std::size_t kSegmentSize = 64 * 1024;
} // namespace

// Test construction of a segment succeeds and reports the host platform
TEST(shared_memory, construction)
{
  shared_memory mem("sm_ctor", kSegmentSize);
  EXPECT_EQ(mem.get_name(), "sm_ctor");
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::host);
}

// Test the `platform` type alias is present and correct, for API
// consistency with the templated resources.
TEST(shared_memory, type_traits)
{
  static_assert(std::is_same_v<shared_memory::platform, umpire::host_platform>,
                "platform should be host_platform");
}

// Test anonymous allocation: allocate(bytes) should succeed (unlike v1's
// HostSharedMemoryResource::allocate(), which always throws), by
// synthesizing a unique internal name.
TEST(shared_memory, anonymous_allocation)
{
  shared_memory mem("sm_anon", kSegmentSize);

  void* ptr = mem.allocate(128);
  ASSERT_NE(ptr, nullptr);

  mem.deallocate(ptr);
}

// Test that two anonymous allocations get distinct storage
TEST(shared_memory, anonymous_allocations_are_distinct)
{
  shared_memory mem("sm_anon2", kSegmentSize);

  void* a = mem.allocate(64);
  void* b = mem.allocate(64);
  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);
  EXPECT_NE(a, b);

  mem.deallocate(a);
  mem.deallocate(b);
}

// Test named allocation: allocate(name, bytes) returns valid storage
TEST(shared_memory, named_allocation)
{
  shared_memory mem("sm_named", kSegmentSize);

  void* ptr = mem.allocate("region_a", 256);
  ASSERT_NE(ptr, nullptr);

  mem.deallocate(ptr);
}

// Test write/read roundtrip through a named allocation
TEST(shared_memory, write_read_roundtrip)
{
  shared_memory mem("sm_rw", kSegmentSize);

  const std::size_t size = 1024;
  void* ptr = mem.allocate("roundtrip", size);
  ASSERT_NE(ptr, nullptr);

  char* bytes = static_cast<char*>(ptr);
  bytes[0] = 'A';
  bytes[size - 1] = 'Z';
  EXPECT_EQ(bytes[0], 'A');
  EXPECT_EQ(bytes[size - 1], 'Z');

  mem.deallocate(ptr);
}

// Test find_pointer_from_name(): found case returns the same pointer as
// the original allocate() call.
TEST(shared_memory, find_pointer_from_name_found)
{
  shared_memory mem("sm_find", kSegmentSize);

  void* ptr = mem.allocate("findable", 128);
  ASSERT_NE(ptr, nullptr);

  void* found = mem.find_pointer_from_name("findable");
  EXPECT_EQ(found, ptr);

  mem.deallocate(ptr);
}

// Test find_pointer_from_name(): not-found case returns nullptr
TEST(shared_memory, find_pointer_from_name_not_found)
{
  shared_memory mem("sm_notfound", kSegmentSize);

  void* found = mem.find_pointer_from_name("does_not_exist");
  EXPECT_EQ(found, nullptr);
}

// Test allocating with an already-used name attaches (ref-counts) rather
// than creating a second block, and returns the same pointer.
TEST(shared_memory, named_allocation_attaches_to_existing)
{
  shared_memory mem("sm_attach", kSegmentSize);

  void* first = mem.allocate("shared_region", 256);
  ASSERT_NE(first, nullptr);

  void* second = mem.allocate("shared_region", 256);
  EXPECT_EQ(second, first);

  // Both references must be released before the underlying block is freed.
  mem.deallocate(first);
  EXPECT_NE(mem.find_pointer_from_name("shared_region"), nullptr);

  mem.deallocate(second);
  EXPECT_EQ(mem.find_pointer_from_name("shared_region"), nullptr);
}

// Test nullptr deallocation is a safe no-op
TEST(shared_memory, nullptr_deallocation_is_safe)
{
  shared_memory mem("sm_null", kSegmentSize);
  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

// Test zero-size anonymous allocation returns nullptr and is a safe no-op
TEST(shared_memory, zero_size_allocation_returns_nullptr)
{
  shared_memory mem("sm_zero", kSegmentSize);

  void* ptr = mem.allocate(std::size_t{0});
  EXPECT_EQ(ptr, nullptr);

  EXPECT_NO_THROW(mem.deallocate(ptr));
}

// Test that an allocation request too large for the segment throws
// out_of_memory_error
TEST(shared_memory, allocation_failure_throws_exception)
{
  shared_memory mem("sm_fail", kSegmentSize);

  EXPECT_THROW(mem.allocate("too_big", kSegmentSize * 4), umpire::out_of_memory_error);
}

// Test registry tracking visibility: a tracked allocation shows up via the
// shared v2 registry, keyed both by pointer and by this resource's id, and
// is removed from the registry once deallocated.
TEST(shared_memory, registry_tracking_visibility)
{
  shared_memory mem("sm_registry", kSegmentSize, /*tracking=*/true);

  auto& registry = umpire::detail::registry::get();
  EXPECT_TRUE(registry.find_allocations_by_memory(&mem).empty());

  void* ptr = mem.allocate("tracked", 512);
  ASSERT_NE(ptr, nullptr);

  EXPECT_TRUE(registry.has_allocation(ptr));

  auto by_ptr = registry.find_allocations_by_memory(&mem);
  ASSERT_EQ(by_ptr.size(), 1u);
  EXPECT_EQ(by_ptr[0].ptr, ptr);
  EXPECT_EQ(by_ptr[0].size, 512u);

  auto by_id = registry.find_allocations_by_memory(mem.get_id());
  EXPECT_EQ(by_id.size(), 1u);

  mem.deallocate(ptr);

  EXPECT_FALSE(registry.has_allocation(ptr));
  EXPECT_TRUE(registry.find_allocations_by_memory(&mem).empty());
}

// Test that disabling tracking avoids any registry interaction
TEST(shared_memory, tracking_disabled_no_registry_interaction)
{
  shared_memory mem("sm_untracked", kSegmentSize, /*tracking=*/false);

  void* ptr = mem.allocate("untracked", 128);
  ASSERT_NE(ptr, nullptr);

  auto& registry = umpire::detail::registry::get();
  EXPECT_FALSE(registry.has_allocation(ptr));
  EXPECT_EQ(mem.get_current_size(), 0u);

  mem.deallocate(ptr);
}

// Test base-class statistics: get_current_size()/get_highwatermark() are
// updated by tracked allocations and deallocations.
TEST(shared_memory, statistics)
{
  shared_memory mem("sm_stats", kSegmentSize);

  void* a = mem.allocate("stat_a", 100);
  EXPECT_EQ(mem.get_current_size(), 100u);
  EXPECT_EQ(mem.get_highwatermark(), 100u);

  void* b = mem.allocate("stat_b", 200);
  EXPECT_EQ(mem.get_current_size(), 300u);
  EXPECT_EQ(mem.get_highwatermark(), 300u);

  mem.deallocate(a);
  EXPECT_EQ(mem.get_current_size(), 200u);
  EXPECT_EQ(mem.get_highwatermark(), 300u);

  mem.deallocate(b);
  EXPECT_EQ(mem.get_current_size(), 0u);
  EXPECT_EQ(mem.get_highwatermark(), 300u);
}

// Test the segment-wide actual-size accounting (distinct from the base
// class's get_actual_size(), which is not used by this resource since
// tracking only calls track_allocation()/untrack_allocation(), not
// update_actual_size() directly -- get_segment_actual_size() reflects the
// shared segment header instead).
TEST(shared_memory, segment_actual_size_accounts_for_overhead)
{
  shared_memory mem("sm_segsize", kSegmentSize);

  std::size_t before = mem.get_segment_actual_size();

  void* ptr = mem.allocate("overhead", 128);
  ASSERT_NE(ptr, nullptr);

  std::size_t after = mem.get_segment_actual_size();
  // Actual segment usage includes block-header and name-string overhead, so
  // it must grow by more than just the requested 128 bytes.
  EXPECT_GT(after, before);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_segment_actual_size(), before);
}

// Test that independent segments (distinct names) do not interfere with
// each other's allocations or statistics.
TEST(shared_memory, independent_instances)
{
  shared_memory mem1("sm_indep1", kSegmentSize);
  shared_memory mem2("sm_indep2", kSegmentSize);

  void* ptr1 = mem1.allocate("a", 100);
  void* ptr2 = mem2.allocate("a", 200);

  EXPECT_EQ(mem1.get_current_size(), 100u);
  EXPECT_EQ(mem2.get_current_size(), 200u);

  mem1.deallocate(ptr1);
  mem2.deallocate(ptr2);
}
