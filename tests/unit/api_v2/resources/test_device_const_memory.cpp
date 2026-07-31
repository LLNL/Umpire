//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/config.hpp"

#include <gtest/gtest.h>
#include <type_traits>

#if defined(UMPIRE_ENABLE_CUDA)

#include "umpire/resource/cuda_device_const_memory.hpp"

#include <cuda_runtime_api.h>

namespace {
bool cuda_available() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  return (error == cudaSuccess && device_count > 0);
}
} // namespace

using namespace umpire::resource;

TEST(cuda_device_const_memory, singleton_returns_same_instance)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  auto& inst1 = cuda_device_const_memory<>::get();
  auto& inst2 = cuda_device_const_memory<>::get();

  EXPECT_EQ(&inst1, &inst2);
  EXPECT_EQ(inst1.get_name(), "DEVICE_CONST");
}

TEST(cuda_device_const_memory, custom_instance_with_custom_name)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_const_memory<> custom("CUSTOM_CONST");
  EXPECT_EQ(custom.get_name(), "CUSTOM_CONST");
}

TEST(cuda_device_const_memory, basic_allocation)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_const_memory<> mem("TEST_CONST");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  mem.deallocate(ptr);
}

TEST(cuda_device_const_memory, sequential_allocations_bump_offset)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_const_memory<> mem("SEQ_CONST");

  void* a = mem.allocate(128);
  void* b = mem.allocate(256);

  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);

  // b must immediately follow a in the bump-offset buffer
  EXPECT_EQ(static_cast<char*>(b), static_cast<char*>(a) + 128);

  // LIFO deallocation order must succeed
  mem.deallocate(b);
  mem.deallocate(a);
}

TEST(cuda_device_const_memory, deallocate_out_of_order_throws)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_const_memory<> mem("OOO_CONST");

  void* a = mem.allocate(128);
  void* b = mem.allocate(256);
  (void)a;

  // Deallocating anything but the most recent allocation must fail.
  EXPECT_THROW(mem.deallocate(a), umpire::runtime_error);

  // Clean up in correct order so later tests aren't polluted.
  mem.deallocate(b);
  mem.deallocate(a);
}

TEST(cuda_device_const_memory, exceeding_max_size_throws)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_const_memory<> mem("OVERFLOW_CONST");

  EXPECT_THROW(mem.allocate(cuda_device_const_max_size + 1), umpire::runtime_error);
}

TEST(cuda_device_const_memory, tracking_enabled_records_allocation)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_const_memory<cuda_device_const_allocator, true> mem("TRACKED_CONST");

  void* ptr = mem.allocate(128);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 128u);
  EXPECT_EQ(mem.get_highwatermark(), 128u);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0u);
}

TEST(cuda_device_const_memory, nullptr_deallocation_is_safe)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_const_memory<> mem("NULL_CONST");
  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

TEST(cuda_device_const_memory, platform_type_is_cuda)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_const_memory<> mem("PLATFORM_CONST");
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::cuda);
}

TEST(cuda_device_const_memory, type_traits)
{
  using tracked_const = cuda_device_const_memory<cuda_device_const_allocator, true>;
  using untracked_const = cuda_device_const_memory<cuda_device_const_allocator, false>;

  static_assert(std::is_same_v<tracked_const::platform, umpire::cuda_platform>,
                "Platform should be cuda_platform");
  static_assert(tracked_const::tracking_enabled == true, "tracking_enabled should be true");
  static_assert(untracked_const::tracking_enabled == false, "tracking_enabled should be false");
}

#endif // UMPIRE_ENABLE_CUDA

#if defined(UMPIRE_ENABLE_HIP)

#include "umpire/resource/hip_device_const_memory.hpp"

#include <hip/hip_runtime.h>

namespace {
bool hip_available() {
  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);
  return (error == hipSuccess && device_count > 0);
}
} // namespace

using namespace umpire::resource;

TEST(hip_device_const_memory, singleton_returns_same_instance)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  auto& inst1 = hip_device_const_memory<>::get();
  auto& inst2 = hip_device_const_memory<>::get();

  EXPECT_EQ(&inst1, &inst2);
  EXPECT_EQ(inst1.get_name(), "DEVICE_CONST");
}

TEST(hip_device_const_memory, basic_allocation)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_const_memory<> mem("TEST_CONST");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  mem.deallocate(ptr);
}

TEST(hip_device_const_memory, sequential_allocations_bump_offset)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_const_memory<> mem("SEQ_CONST");

  void* a = mem.allocate(128);
  void* b = mem.allocate(256);

  ASSERT_NE(a, nullptr);
  ASSERT_NE(b, nullptr);

  EXPECT_EQ(static_cast<char*>(b), static_cast<char*>(a) + 128);

  mem.deallocate(b);
  mem.deallocate(a);
}

TEST(hip_device_const_memory, deallocate_out_of_order_throws)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_const_memory<> mem("OOO_CONST");

  void* a = mem.allocate(128);
  void* b = mem.allocate(256);

  EXPECT_THROW(mem.deallocate(a), umpire::runtime_error);

  mem.deallocate(b);
  mem.deallocate(a);
}

TEST(hip_device_const_memory, exceeding_max_size_throws)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_const_memory<> mem("OVERFLOW_CONST");

  EXPECT_THROW(mem.allocate(hip_device_const_max_size + 1), umpire::runtime_error);
}

TEST(hip_device_const_memory, platform_type_is_hip)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_const_memory<> mem("PLATFORM_CONST");
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::hip);
}

TEST(hip_device_const_memory, type_traits)
{
  using tracked_const = hip_device_const_memory<hip_device_const_allocator, true>;
  using untracked_const = hip_device_const_memory<hip_device_const_allocator, false>;

  static_assert(std::is_same_v<tracked_const::platform, umpire::hip_platform>,
                "Platform should be hip_platform");
  static_assert(tracked_const::tracking_enabled == true, "tracking_enabled should be true");
  static_assert(untracked_const::tracking_enabled == false, "tracking_enabled should be false");
}

#endif // UMPIRE_ENABLE_HIP

// Keep this translation unit non-empty even when no constant-memory-capable
// backend is enabled.
#if !defined(UMPIRE_ENABLE_CUDA) && !defined(UMPIRE_ENABLE_HIP)
TEST(device_const_memory, no_backend_enabled)
{
  GTEST_SKIP() << "No constant-memory-capable backend (CUDA/HIP) enabled in this build";
}
#endif
