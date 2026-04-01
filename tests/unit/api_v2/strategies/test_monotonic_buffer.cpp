//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/monotonic_buffer.hpp"
#include "umpire/resource/host_memory.hpp"
#include "umpire/memory.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace {

class test_memory : public umpire::memory {
public:
  using platform = umpire::host_platform;

  test_memory()
    : umpire::memory{"test_parent"}
  { }

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

  umpire::resource::Platform get_platform() const override
  {
    return umpire::resource::Platform::host;
  }
};

} // namespace

TEST(monotonic_buffer, construct_with_valid_parameters)
{
  test_memory parent;
  umpire::strategy::monotonic_buffer<test_memory> buffer("monotonic", &parent, 1024);

  EXPECT_EQ(buffer.get_parent(), &parent);
  EXPECT_EQ(buffer.get_name(), "monotonic");
  EXPECT_EQ(buffer.get_capacity(), 1024);
  EXPECT_EQ(buffer.get_current_size(), 0);
  EXPECT_EQ(buffer.get_high_watermark(), 0);
}

TEST(monotonic_buffer, construct_with_nullptr_throws)
{
  EXPECT_THROW(
    umpire::strategy::monotonic_buffer<test_memory> buffer("monotonic", nullptr, 1024),
    std::invalid_argument
  );
}

TEST(monotonic_buffer, construct_with_zero_capacity_throws)
{
  test_memory parent;
  EXPECT_THROW(
    umpire::strategy::monotonic_buffer<test_memory> buffer("monotonic", &parent, 0),
    std::invalid_argument
  );
}

TEST(monotonic_buffer, get_platform_delegates_to_parent)
{
  test_memory parent;
  umpire::strategy::monotonic_buffer<test_memory> buffer("monotonic", &parent, 1024);

  EXPECT_EQ(buffer.get_platform(), parent.get_platform());
  EXPECT_EQ(buffer.get_platform(), umpire::resource::Platform::host);
}

TEST(monotonic_buffer, basic_allocation)
{
  test_memory parent;
  umpire::strategy::monotonic_buffer<test_memory> buffer("monotonic", &parent, 1024);

  void* ptr = buffer.allocate(64);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(buffer.get_current_size(), 64);
  EXPECT_EQ(buffer.get_high_watermark(), 64);
}

TEST(monotonic_buffer, zero_size_allocation_returns_nullptr)
{
  test_memory parent;
  umpire::strategy::monotonic_buffer<test_memory> buffer("monotonic", &parent, 1024);

  EXPECT_EQ(buffer.allocate(0), nullptr);
  EXPECT_EQ(buffer.get_current_size(), 0);
}

TEST(monotonic_buffer, allocations_bump_forward)
{
  test_memory parent;
  umpire::strategy::monotonic_buffer<test_memory> buffer("monotonic", &parent, 1024);

  void* ptr1 = buffer.allocate(32);
  void* ptr2 = buffer.allocate(64);

  ASSERT_NE(ptr1, nullptr);
  ASSERT_NE(ptr2, nullptr);
  EXPECT_LT(reinterpret_cast<std::uintptr_t>(ptr1), reinterpret_cast<std::uintptr_t>(ptr2));
  EXPECT_GE(reinterpret_cast<std::uintptr_t>(ptr2) - reinterpret_cast<std::uintptr_t>(ptr1), 32u);
}

TEST(monotonic_buffer, allocations_preserve_alignment)
{
  test_memory parent;
  umpire::strategy::monotonic_buffer<test_memory> buffer("monotonic", &parent, 1024);

  void* ptr1 = buffer.allocate(1);
  void* ptr2 = buffer.allocate(1);

  ASSERT_NE(ptr1, nullptr);
  ASSERT_NE(ptr2, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr1) % alignof(std::max_align_t), 0u);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr2) % alignof(std::max_align_t), 0u);
}

TEST(monotonic_buffer, exhaustion_throws_without_modifying_usage)
{
  test_memory parent;
  umpire::strategy::monotonic_buffer<test_memory> buffer("monotonic", &parent, 64);

  void* ptr = buffer.allocate(32);
  ASSERT_NE(ptr, nullptr);
  const std::size_t used_before = buffer.get_current_size();
  const std::size_t high_watermark_before = buffer.get_high_watermark();

  EXPECT_THROW(buffer.allocate(64), umpire::runtime_error);
  EXPECT_EQ(buffer.get_current_size(), used_before);
  EXPECT_EQ(buffer.get_high_watermark(), high_watermark_before);
}

TEST(monotonic_buffer, deallocate_is_noop)
{
  test_memory parent;
  umpire::strategy::monotonic_buffer<test_memory> buffer("monotonic", &parent, 1024);

  void* ptr = buffer.allocate(64);
  ASSERT_NE(ptr, nullptr);

  buffer.deallocate(ptr);
  EXPECT_EQ(buffer.get_current_size(), 64);
  EXPECT_EQ(buffer.get_high_watermark(), 64);
}

TEST(monotonic_buffer, deallocate_nullptr_is_safe)
{
  test_memory parent;
  umpire::strategy::monotonic_buffer<test_memory> buffer("monotonic", &parent, 1024);

  EXPECT_NO_THROW(buffer.deallocate(nullptr));
  EXPECT_EQ(buffer.get_current_size(), 0);
}

TEST(monotonic_buffer, release_resets_buffer_for_reuse)
{
  test_memory parent;
  umpire::strategy::monotonic_buffer<test_memory> buffer("monotonic", &parent, 1024);

  void* ptr1 = buffer.allocate(64);
  void* ptr2 = buffer.allocate(128);
  ASSERT_NE(ptr1, nullptr);
  ASSERT_NE(ptr2, nullptr);
  EXPECT_GT(buffer.get_current_size(), 0);

  buffer.release();

  EXPECT_EQ(buffer.get_current_size(), 0);
  EXPECT_GE(buffer.get_high_watermark(), 192);

  void* ptr3 = buffer.allocate(64);
  EXPECT_EQ(ptr3, ptr1);
}

TEST(monotonic_buffer, remaining_capacity_tracks_alignment)
{
  test_memory parent;
  umpire::strategy::monotonic_buffer<test_memory> buffer("monotonic", &parent, 128);

  EXPECT_EQ(buffer.get_remaining_capacity(), 128);

  buffer.allocate(1);
  EXPECT_LT(buffer.get_remaining_capacity(), 128);
}

TEST(monotonic_buffer, platform_type_propagation)
{
  using host_mem = umpire::resource::host_memory<>;
  using monotonic_host = umpire::strategy::monotonic_buffer<host_mem>;

  static_assert(std::is_same<monotonic_host::platform, umpire::host_platform>::value,
                "Platform type should be propagated from wrapped memory");
}

TEST(monotonic_buffer, composition_with_host_memory)
{
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::monotonic_buffer<umpire::resource::host_memory<>>
    buffer("host_monotonic", &host, 1024);

  void* ptr = buffer.allocate(128);
  ASSERT_NE(ptr, nullptr);

  char* bytes = static_cast<char*>(ptr);
  bytes[0] = 'A';
  bytes[127] = 'Z';
  EXPECT_EQ(bytes[0], 'A');
  EXPECT_EQ(bytes[127], 'Z');
}
