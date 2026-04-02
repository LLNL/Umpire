//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)

#include "umpire/resource/openmp_target_memory.hpp"

#include <gtest/gtest.h>
#include <omp.h>

#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace umpire::resource;

namespace {

bool openmp_target_available()
{
  return omp_get_num_devices() > 0;
}

int openmp_target_device()
{
  return omp_get_default_device();
}

} // namespace

#define SKIP_IF_NO_OPENMP_TARGET_DEVICE() \
  do { \
    if (!openmp_target_available()) { \
      GTEST_SKIP() << "OpenMP target device not available"; \
    } \
  } while (false)

TEST(openmp_target_memory, singleton_returns_same_instance)
{
  SKIP_IF_NO_OPENMP_TARGET_DEVICE();

  auto& inst1 = openmp_target_memory<>::get();
  auto& inst2 = openmp_target_memory<>::get();

  EXPECT_EQ(&inst1, &inst2);
  EXPECT_EQ(inst1.get_name(), "OMP_TARGET");
  EXPECT_EQ(inst1.get_device_id(), openmp_target_device());
}

TEST(openmp_target_memory, custom_instance_with_custom_name)
{
  SKIP_IF_NO_OPENMP_TARGET_DEVICE();

  openmp_target_memory<> custom("CUSTOM_OMP_TARGET", openmp_target_device());
  EXPECT_EQ(custom.get_name(), "CUSTOM_OMP_TARGET");
  EXPECT_EQ(custom.get_device_id(), openmp_target_device());
}

TEST(openmp_target_memory, custom_instance_with_device_id)
{
  SKIP_IF_NO_OPENMP_TARGET_DEVICE();

  openmp_target_memory<> custom(openmp_target_device());
  EXPECT_EQ(custom.get_name(), "OMP_TARGET_" + std::to_string(openmp_target_device()));
  EXPECT_EQ(custom.get_device_id(), openmp_target_device());
}

TEST(openmp_target_memory, basic_allocation)
{
  SKIP_IF_NO_OPENMP_TARGET_DEVICE();

  openmp_target_memory<> mem("TEST_OMP_TARGET", openmp_target_device());
  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  std::vector<char> host_data(1024);
  host_data.front() = 'A';
  host_data.back() = 'Z';

  ASSERT_EQ(omp_target_memcpy(ptr,
                              host_data.data(),
                              host_data.size(),
                              0,
                              0,
                              openmp_target_device(),
                              omp_get_initial_device()),
            0);

  std::vector<char> readback(1024, 0);
  ASSERT_EQ(omp_target_memcpy(readback.data(),
                              ptr,
                              readback.size(),
                              0,
                              0,
                              omp_get_initial_device(),
                              openmp_target_device()),
            0);

  EXPECT_EQ(readback.front(), 'A');
  EXPECT_EQ(readback.back(), 'Z');

  mem.deallocate(ptr);
}

TEST(openmp_target_memory, tracking_enabled_records_allocation)
{
  SKIP_IF_NO_OPENMP_TARGET_DEVICE();

  openmp_target_memory<omp_target_allocator, true> mem("TRACKED_OMP_TARGET", openmp_target_device());
  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 1024);
  EXPECT_EQ(mem.get_highwatermark(), 1024);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 1024);
}

TEST(openmp_target_memory, tracking_disabled_no_registry_interaction)
{
  SKIP_IF_NO_OPENMP_TARGET_DEVICE();

  openmp_target_memory<omp_target_allocator, false> mem("UNTRACKED_OMP_TARGET", openmp_target_device());
  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 0);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 0);
}

TEST(openmp_target_memory, zero_size_allocation_returns_nullptr)
{
  SKIP_IF_NO_OPENMP_TARGET_DEVICE();

  openmp_target_memory<> mem("ZERO_OMP_TARGET", openmp_target_device());
  void* ptr = mem.allocate(0);
  EXPECT_EQ(ptr, nullptr);
  mem.deallocate(ptr);
}

TEST(openmp_target_memory, nullptr_deallocation_is_safe)
{
  SKIP_IF_NO_OPENMP_TARGET_DEVICE();

  openmp_target_memory<> mem("NULL_OMP_TARGET", openmp_target_device());
  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

TEST(openmp_target_memory, multiple_allocations_correct_statistics)
{
  SKIP_IF_NO_OPENMP_TARGET_DEVICE();

  openmp_target_memory<> mem("MULTI_OMP_TARGET", openmp_target_device());

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

TEST(openmp_target_memory, allocation_failure_throws_exception)
{
  SKIP_IF_NO_OPENMP_TARGET_DEVICE();

  openmp_target_memory<> mem("FAIL_OMP_TARGET", openmp_target_device());
  std::size_t huge_size = std::numeric_limits<std::size_t>::max() - 1024;
  EXPECT_THROW(mem.allocate(huge_size), umpire::out_of_memory_error);
}

TEST(openmp_target_memory, platform_type_is_omp_target)
{
  SKIP_IF_NO_OPENMP_TARGET_DEVICE();

  openmp_target_memory<> mem("PLATFORM_OMP_TARGET", openmp_target_device());
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::omp_target);
}

TEST(openmp_target_memory, convenience_aliases)
{
  static_assert(default_openmp_target_memory::tracking_enabled == true,
                "default_openmp_target_memory should have tracking enabled");
  static_assert(fast_openmp_target_memory::tracking_enabled == false,
                "fast_openmp_target_memory should have tracking disabled");
}

TEST(openmp_target_memory, type_traits)
{
  using tracked_omp_target = openmp_target_memory<omp_target_allocator, true>;
  using untracked_omp_target = openmp_target_memory<omp_target_allocator, false>;

  static_assert(std::is_same_v<tracked_omp_target::platform, umpire::omp_target_platform>,
                "Platform should be omp_target_platform");
  static_assert(std::is_same_v<tracked_omp_target::allocator_type, omp_target_allocator>,
                "Allocator type should be omp_target_allocator");
  static_assert(tracked_omp_target::tracking_enabled == true,
                "tracking_enabled should be true");
  static_assert(untracked_omp_target::tracking_enabled == false,
                "tracking_enabled should be false");
}

TEST(openmp_target_memory, independent_instances)
{
  SKIP_IF_NO_OPENMP_TARGET_DEVICE();

  openmp_target_memory<> mem1("OMP_TARGET_ONE", openmp_target_device());
  openmp_target_memory<> mem2("OMP_TARGET_TWO", openmp_target_device());

  void* ptr1 = mem1.allocate(100);
  void* ptr2 = mem2.allocate(200);

  EXPECT_EQ(mem1.get_current_size(), 100);
  EXPECT_EQ(mem2.get_current_size(), 200);

  mem1.deallocate(ptr1);
  mem2.deallocate(ptr2);
}

#endif
