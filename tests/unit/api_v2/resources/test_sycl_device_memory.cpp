//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_SYCL)

#include "umpire/resource/sycl_device_memory.hpp"

#include <gtest/gtest.h>

#include <array>
#include <type_traits>
#include <vector>

namespace {

std::vector<sycl::device> sycl_gpu_devices()
{
  try {
    return sycl::device::get_devices(sycl::info::device_type::gpu);
  } catch (...) {
    return {};
  }
}

bool sycl_gpu_available()
{
  return !sycl_gpu_devices().empty();
}

sycl::queue make_test_queue()
{
  return sycl::queue{sycl_gpu_devices().front()};
}

} // namespace

using namespace umpire::resource;

TEST(sycl_device_memory, custom_instance_with_queue)
{
  if (!sycl_gpu_available()) GTEST_SKIP() << "SYCL GPU device not available";

  auto queue = make_test_queue();
  sycl_device_memory<> mem("SYCL_GPU", queue);

  EXPECT_EQ(mem.get_name(), "SYCL_GPU");
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::sycl);
  EXPECT_EQ(mem.get_queue().get_device(), queue.get_device());
}

TEST(sycl_device_memory, basic_allocation)
{
  if (!sycl_gpu_available()) GTEST_SKIP() << "SYCL GPU device not available";

  auto queue = make_test_queue();
  sycl_device_memory<> mem("TEST_SYCL", queue);

  auto* ptr = static_cast<int*>(mem.allocate(4 * sizeof(int)));
  ASSERT_NE(ptr, nullptr);

  std::array<int, 4> host_values{{2, 4, 6, 8}};
  std::array<int, 4> readback{{0, 0, 0, 0}};

  queue.memcpy(ptr, host_values.data(), sizeof(host_values)).wait();
  queue.memcpy(readback.data(), ptr, sizeof(readback)).wait();

  EXPECT_EQ(readback, host_values);

  mem.deallocate(ptr);
}

TEST(sycl_device_memory, tracking_enabled_records_allocation)
{
  if (!sycl_gpu_available()) GTEST_SKIP() << "SYCL GPU device not available";

  sycl_device_memory<sycl_default_allocator, true> mem("TRACKED_SYCL", make_test_queue());

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 1024);
  EXPECT_EQ(mem.get_highwatermark(), 1024);

  mem.deallocate(ptr);

  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 1024);
}

TEST(sycl_device_memory, tracking_disabled_no_registry_interaction)
{
  if (!sycl_gpu_available()) GTEST_SKIP() << "SYCL GPU device not available";

  sycl_device_memory<sycl_default_allocator, false> mem("UNTRACKED_SYCL", make_test_queue());

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 0);

  mem.deallocate(ptr);

  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 0);
}

TEST(sycl_device_memory, zero_size_allocation_returns_nullptr)
{
  if (!sycl_gpu_available()) GTEST_SKIP() << "SYCL GPU device not available";

  sycl_device_memory<> mem("ZERO_SYCL", make_test_queue());

  void* ptr = mem.allocate(0);
  EXPECT_EQ(ptr, nullptr);
  mem.deallocate(ptr);
}

TEST(sycl_device_memory, nullptr_deallocation_is_safe)
{
  if (!sycl_gpu_available()) GTEST_SKIP() << "SYCL GPU device not available";

  sycl_device_memory<> mem("NULL_SYCL", make_test_queue());

  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

TEST(sycl_device_memory, type_traits)
{
  using tracked_sycl = sycl_device_memory<sycl_default_allocator, true>;
  using untracked_sycl = sycl_device_memory<sycl_default_allocator, false>;

  static_assert(std::is_same_v<tracked_sycl::platform, umpire::sycl_platform>,
                "Platform should be sycl_platform");
  static_assert(std::is_same_v<tracked_sycl::allocator_type, sycl_default_allocator>,
                "Allocator type should be sycl_default_allocator");
  static_assert(tracked_sycl::tracking_enabled == true,
                "tracking_enabled should be true");
  static_assert(untracked_sycl::tracking_enabled == false,
                "tracking_enabled should be false");
}

#endif
