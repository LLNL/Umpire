//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/size_limiter.hpp"
#include "umpire/resource/host_memory.hpp"
#include "umpire/memory.hpp"

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

class throwing_memory : public umpire::memory {
public:
  using platform = umpire::host_platform;

  throwing_memory()
    : umpire::memory{"throwing_parent"}
  {
  }

  void* allocate(std::size_t) override
  {
    throw std::runtime_error("Intentional allocation failure");
  }

  void deallocate(void*) override
  {
    throw std::runtime_error("Intentional deallocation failure");
  }

  umpire::resource::Platform get_platform() const override
  {
    return umpire::resource::Platform::host;
  }
};

class deallocate_throwing_memory : public umpire::memory {
public:
  using platform = umpire::host_platform;

  deallocate_throwing_memory()
    : umpire::memory{"deallocate_throwing_parent"}
  {
  }

  void* allocate(std::size_t size) override
  {
    void* ptr{std::malloc(size)};
    track_allocation(ptr, size);
    return ptr;
  }

  void deallocate(void*) override
  {
    throw std::runtime_error("Intentional deallocation failure");
  }

  umpire::resource::Platform get_platform() const override
  {
    return umpire::resource::Platform::host;
  }
};

} // namespace

TEST(size_limiter, construct_with_valid_parent)
{
  test_memory parent;
  umpire::strategy::size_limiter<test_memory> limiter("limiter", &parent, 1024);

  EXPECT_EQ(limiter.get_parent(), &parent);
  EXPECT_EQ(limiter.get_name(), "limiter");
  EXPECT_EQ(limiter.get_limit(), 1024);
  EXPECT_EQ(limiter.get_current_usage(), 0);
}

TEST(size_limiter, construct_with_nullptr_throws)
{
  EXPECT_THROW(
    umpire::strategy::size_limiter<test_memory> limiter("limiter", nullptr, 1024),
    std::invalid_argument);
}

TEST(size_limiter, get_platform_delegates_to_parent)
{
  test_memory parent;
  umpire::strategy::size_limiter<test_memory> limiter("limiter", &parent, 1024);

  EXPECT_EQ(limiter.get_platform(), parent.get_platform());
  EXPECT_EQ(limiter.get_platform(), umpire::resource::Platform::host);
}

TEST(size_limiter, allocate_within_limit)
{
  test_memory parent;
  umpire::strategy::size_limiter<test_memory> limiter("limiter", &parent, 256);

  void* ptr = limiter.allocate(128);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(limiter.get_current_usage(), 128);
  EXPECT_EQ(parent.get_current_size(), 128);

  limiter.deallocate(ptr);
  EXPECT_EQ(limiter.get_current_usage(), 0);
  EXPECT_EQ(parent.get_current_size(), 0);
}

TEST(size_limiter, multiple_allocations_track_live_usage)
{
  test_memory parent;
  umpire::strategy::size_limiter<test_memory> limiter("limiter", &parent, 512);

  void* ptr1 = limiter.allocate(64);
  void* ptr2 = limiter.allocate(128);
  void* ptr3 = limiter.allocate(32);

  EXPECT_EQ(limiter.get_current_usage(), 224);

  limiter.deallocate(ptr2);
  EXPECT_EQ(limiter.get_current_usage(), 96);

  limiter.deallocate(ptr1);
  limiter.deallocate(ptr3);
  EXPECT_EQ(limiter.get_current_usage(), 0);
}

TEST(size_limiter, limit_exceeded_throws_logic_error)
{
  test_memory parent;
  umpire::strategy::size_limiter<test_memory> limiter("limiter", &parent, 128);

  void* ptr = limiter.allocate(64);
  ASSERT_NE(ptr, nullptr);

  EXPECT_THROW(limiter.allocate(80), std::logic_error);
  EXPECT_EQ(limiter.get_current_usage(), 64);
  EXPECT_EQ(parent.get_current_size(), 64);

  limiter.deallocate(ptr);
}

TEST(size_limiter, allocation_failure_rolls_back_usage)
{
  throwing_memory parent;
  umpire::strategy::size_limiter<throwing_memory> limiter("limiter", &parent, 128);

  EXPECT_THROW(limiter.allocate(64), std::runtime_error);
  EXPECT_EQ(limiter.get_current_usage(), 0);
}

TEST(size_limiter, zero_size_allocation_delegates_without_usage)
{
  test_memory parent;
  umpire::strategy::size_limiter<test_memory> limiter("limiter", &parent, 128);

  void* ptr = limiter.allocate(0);
  EXPECT_EQ(limiter.get_current_usage(), 0);
  limiter.deallocate(ptr);
  EXPECT_EQ(limiter.get_current_usage(), 0);
}

TEST(size_limiter, nullptr_deallocation_is_safe)
{
  test_memory parent;
  umpire::strategy::size_limiter<test_memory> limiter("limiter", &parent, 128);

  EXPECT_NO_THROW(limiter.deallocate(nullptr));
  EXPECT_EQ(limiter.get_current_usage(), 0);
}

TEST(size_limiter, unknown_pointer_throws_runtime_error)
{
  test_memory parent;
  umpire::strategy::size_limiter<test_memory> limiter("limiter", &parent, 128);

  int stack_value = 42;
  EXPECT_THROW(limiter.deallocate(&stack_value), std::runtime_error);
  EXPECT_EQ(limiter.get_current_usage(), 0);
}

TEST(size_limiter, current_usage_not_reduced_when_parent_deallocate_throws)
{
  deallocate_throwing_memory parent;
  umpire::strategy::size_limiter<deallocate_throwing_memory> limiter("limiter", &parent, 128);

  void* ptr = limiter.allocate(32);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(limiter.get_current_usage(), 32);
  EXPECT_THROW(limiter.deallocate(ptr), std::runtime_error);
  EXPECT_EQ(limiter.get_current_usage(), 32);
}

TEST(size_limiter, composition_with_host_memory)
{
  auto& host = umpire::resource::host_memory<>::get();
  umpire::strategy::size_limiter<umpire::resource::host_memory<>> limiter("host_limiter", &host, 1024);

  void* ptr = limiter.allocate(256);
  ASSERT_NE(ptr, nullptr);

  char* bytes = static_cast<char*>(ptr);
  bytes[0] = 'A';
  bytes[255] = 'Z';
  EXPECT_EQ(bytes[0], 'A');
  EXPECT_EQ(bytes[255], 'Z');

  limiter.deallocate(ptr);
  EXPECT_EQ(limiter.get_current_usage(), 0);
}

TEST(size_limiter, platform_type_propagation)
{
  using host_mem = umpire::resource::host_memory<>;
  using limited_host = umpire::strategy::size_limiter<host_mem>;

  static_assert(std::is_same<limited_host::platform, umpire::host_platform>::value,
                "Platform type should be propagated from wrapped memory");
}
