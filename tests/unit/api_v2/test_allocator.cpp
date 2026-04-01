//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/allocator.hpp"
#include "umpire/memory.hpp"
#include "umpire/resource/host_memory.hpp"

#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

namespace {

class instrumented_memory : public umpire::memory {
public:
  using platform = umpire::host_platform;

  instrumented_memory()
    : umpire::memory{"instrumented_memory"}
  {
  }

  void* allocate(std::size_t size) override
  {
    ++allocation_calls_;
    last_allocation_size_ = size;

    void* ptr = std::malloc(size);
    track_allocation(ptr, size);
    return ptr;
  }

  void deallocate(void* ptr) override
  {
    ++deallocation_calls_;
    last_deallocation_ptr_ = ptr;

    if (ptr) {
      untrack_allocation(ptr);
      std::free(ptr);
    }
  }

  umpire::resource::Platform get_platform() const override
  {
    return umpire::resource::Platform::host;
  }

  int allocation_calls_{0};
  int deallocation_calls_{0};
  std::size_t last_allocation_size_{0};
  void* last_deallocation_ptr_{nullptr};
};

} // namespace

TEST(allocator, construct_with_valid_memory)
{
  instrumented_memory memory;
  umpire::allocator<int, instrumented_memory> alloc(&memory);

  EXPECT_EQ(alloc.get_memory(), &memory);
  EXPECT_EQ(alloc.get_id(), memory.get_id());
  EXPECT_EQ(alloc.get_name(), memory.get_name());
}

TEST(allocator, construct_with_nullptr_throws)
{
  EXPECT_THROW((umpire::allocator<int, instrumented_memory>{nullptr}), std::invalid_argument);
}

TEST(allocator, platform_type_propagation)
{
  using host_allocator = umpire::allocator<int, umpire::resource::host_memory<>>;
  static_assert(std::is_same_v<host_allocator::platform, umpire::host_platform>,
                "Allocator should propagate the wrapped memory platform");
}

TEST(allocator, allocate_and_deallocate_delegate_to_memory)
{
  instrumented_memory memory;
  umpire::allocator<int, instrumented_memory> alloc(&memory);

  int* ptr = alloc.allocate(4);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(memory.allocation_calls_, 1);
  EXPECT_EQ(memory.last_allocation_size_, 4 * sizeof(int));

  alloc.deallocate(ptr, 4);
  EXPECT_EQ(memory.deallocation_calls_, 1);
  EXPECT_EQ(memory.last_deallocation_ptr_, ptr);
}

TEST(allocator, introspection_reports_element_counts)
{
  instrumented_memory memory;
  umpire::allocator<int, instrumented_memory> alloc(&memory);

  int* first = alloc.allocate(3);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(alloc.get_current_size(), 3u);
  EXPECT_EQ(alloc.get_actual_size(), 3u);
  EXPECT_EQ(alloc.get_highwatermark(), 3u);

  int* second = alloc.allocate(5);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(alloc.get_current_size(), 8u);
  EXPECT_EQ(alloc.get_actual_size(), 8u);
  EXPECT_EQ(alloc.get_highwatermark(), 8u);

  alloc.deallocate(first, 3);
  EXPECT_EQ(alloc.get_current_size(), 5u);
  EXPECT_EQ(alloc.get_highwatermark(), 8u);

  alloc.deallocate(second, 5);
  EXPECT_EQ(alloc.get_current_size(), 0u);
  EXPECT_EQ(alloc.get_highwatermark(), 8u);
}

TEST(allocator, comparison_operators_use_underlying_memory_identity)
{
  instrumented_memory first_memory;
  instrumented_memory second_memory;

  umpire::allocator<int, instrumented_memory> first(&first_memory);
  umpire::allocator<int, instrumented_memory> same(&first_memory);
  umpire::allocator<int, instrumented_memory> different(&second_memory);
  umpire::allocator<char, instrumented_memory> rebound(&first_memory);

  EXPECT_TRUE(first == same);
  EXPECT_FALSE(first != same);

  EXPECT_TRUE(first == rebound);
  EXPECT_FALSE(first != rebound);

  EXPECT_FALSE(first == different);
  EXPECT_TRUE(first != different);
}

TEST(allocator, vector_integration)
{
  auto& host = umpire::resource::host_memory<>::get();
  using alloc_type = umpire::allocator<int, umpire::resource::host_memory<>>;

  std::vector<int, alloc_type> values{alloc_type{&host}};
  values.resize(8);

  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<int>(i * 2);
  }

  EXPECT_EQ(values.front(), 0);
  EXPECT_EQ(values.back(), 14);
}

TEST(allocator, map_integration)
{
  auto& host = umpire::resource::host_memory<>::get();
  using value_type = std::pair<const int, std::string>;
  using alloc_type = umpire::allocator<value_type, umpire::resource::host_memory<>>;

  std::map<int, std::string, std::less<int>, alloc_type> values{alloc_type{&host}};
  values.emplace(1, "one");
  values.emplace(2, "two");

  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(values.at(1), "one");
  EXPECT_EQ(values.at(2), "two");
}

TEST(allocator, unordered_map_integration)
{
  auto& host = umpire::resource::host_memory<>::get();
  using value_type = std::pair<const int, std::string>;
  using alloc_type = umpire::allocator<value_type, umpire::resource::host_memory<>>;

  std::unordered_map<int, std::string, std::hash<int>, std::equal_to<int>, alloc_type> values{
    0, std::hash<int>{}, std::equal_to<int>{}, alloc_type{&host}};
  values.emplace(3, "three");
  values.emplace(4, "four");

  ASSERT_EQ(values.size(), 2u);
  EXPECT_EQ(values.at(3), "three");
  EXPECT_EQ(values.at(4), "four");
}

TEST(allocator, allocate_shared_uses_rebind)
{
  auto& host = umpire::resource::host_memory<>::get();
  using alloc_type = umpire::allocator<int, umpire::resource::host_memory<>>;

  auto value = std::allocate_shared<int>(alloc_type{&host}, 42);
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, 42);
}
