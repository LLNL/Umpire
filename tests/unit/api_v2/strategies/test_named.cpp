//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/named.hpp"
#include "umpire/detail/registry.hpp"
#include "umpire/memory.hpp"
#include "umpire/resource/host_memory.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace {

class test_memory : public umpire::memory {
public:
  using platform = umpire::host_platform;

  test_memory()
    : umpire::memory{"test_parent"}
  {
  }

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

class instrumented_memory : public umpire::memory {
public:
  using platform = umpire::host_platform;

  instrumented_memory()
    : umpire::memory{"instrumented_parent"}
  {
  }

  void* allocate(std::size_t size) override
  {
    ++allocation_calls_;
    last_allocation_size_ = size;

    void* ptr{std::malloc(size)};
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

TEST(named, construct_with_valid_parent)
{
  test_memory parent;
  umpire::strategy::named<test_memory> strategy("meaningful_name", &parent);

  EXPECT_EQ(strategy.get_parent(), &parent);
  EXPECT_EQ(strategy.get_name(), "meaningful_name");
}

TEST(named, construct_with_nullptr_throws)
{
  EXPECT_THROW(
    umpire::strategy::named<test_memory> strategy("meaningful_name", nullptr),
    std::invalid_argument);
}

TEST(named, registers_allocator_by_custom_name)
{
  test_memory parent;
  umpire::strategy::named<test_memory> strategy("registry_name", &parent);

  auto* found = umpire::detail::registry::get().find_allocator_by_name("registry_name");
  EXPECT_EQ(found, &strategy);
}

TEST(named, get_platform_delegates_to_parent)
{
  test_memory parent;
  umpire::strategy::named<test_memory> strategy("meaningful_name", &parent);

  EXPECT_EQ(strategy.get_platform(), parent.get_platform());
  EXPECT_EQ(strategy.get_platform(), umpire::resource::Platform::host);
}

TEST(named, allocate_delegates_to_parent)
{
  instrumented_memory parent;
  umpire::strategy::named<instrumented_memory> strategy("named_wrapper", &parent);

  void* ptr = strategy.allocate(128);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(parent.allocation_calls_, 1);
  EXPECT_EQ(parent.last_allocation_size_, 128);

  strategy.deallocate(ptr);
  EXPECT_EQ(parent.deallocation_calls_, 1);
  EXPECT_EQ(parent.last_deallocation_ptr_, ptr);
}

TEST(named, nullptr_deallocation_is_forwarded_safely)
{
  instrumented_memory parent;
  umpire::strategy::named<instrumented_memory> strategy("named_wrapper", &parent);

  EXPECT_NO_THROW(strategy.deallocate(nullptr));
  EXPECT_EQ(parent.deallocation_calls_, 1);
  EXPECT_EQ(parent.last_deallocation_ptr_, nullptr);
}

TEST(named, composition_with_host_memory)
{
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::named<umpire::resource::host_memory<>> strategy("HOST_NAMED", &host);

  EXPECT_EQ(strategy.get_name(), "HOST_NAMED");

  void* ptr = strategy.allocate(256);
  ASSERT_NE(ptr, nullptr);

  char* bytes = static_cast<char*>(ptr);
  bytes[0] = 'A';
  bytes[255] = 'Z';
  EXPECT_EQ(bytes[0], 'A');
  EXPECT_EQ(bytes[255], 'Z');

  strategy.deallocate(ptr);
}

TEST(named, platform_type_propagation)
{
  using host_mem = umpire::resource::host_memory<>;
  using named_host = umpire::strategy::named<host_mem>;

  static_assert(std::is_same<named_host::platform, umpire::host_platform>::value,
                "Platform type should be propagated from wrapped memory");
}
