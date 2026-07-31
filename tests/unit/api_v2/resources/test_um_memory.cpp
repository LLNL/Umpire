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

#include "umpire/resource/cuda_um_memory.hpp"

#include <cuda_runtime_api.h>

namespace {
bool cuda_available() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  return (error == cudaSuccess && device_count > 0);
}
} // namespace

using namespace umpire::resource;

TEST(cuda_um_memory, singleton_returns_same_instance)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  auto& inst1 = cuda_um_memory<>::get();
  auto& inst2 = cuda_um_memory<>::get();

  EXPECT_EQ(&inst1, &inst2);
  EXPECT_EQ(inst1.get_name(), "UM");
}

TEST(cuda_um_memory, custom_instance_with_custom_name)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_um_memory<> custom("CUSTOM_UM");
  EXPECT_EQ(custom.get_name(), "CUSTOM_UM");
}

TEST(cuda_um_memory, basic_allocation)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_um_memory<> mem("TEST_UM");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  // Unified memory is host-accessible directly
  char* cptr = static_cast<char*>(ptr);
  cptr[0] = 'A';
  cptr[1023] = 'Z';

  EXPECT_EQ(cptr[0], 'A');
  EXPECT_EQ(cptr[1023], 'Z');

  mem.deallocate(ptr);
}

TEST(cuda_um_memory, tracking_enabled_records_allocation)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_um_memory<cuda_um_allocator, true> mem("TRACKED_UM");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 1024u);
  EXPECT_EQ(mem.get_highwatermark(), 1024u);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0u);
  EXPECT_EQ(mem.get_highwatermark(), 1024u);
}

TEST(cuda_um_memory, tracking_disabled_no_registry_interaction)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_um_memory<cuda_um_allocator, false> mem("UNTRACKED_UM");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 0u);
  EXPECT_EQ(mem.get_highwatermark(), 0u);

  mem.deallocate(ptr);
}

TEST(cuda_um_memory, zero_size_allocation_returns_nullptr)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_um_memory<> mem("ZERO_UM");

  void* ptr = mem.allocate(0);
  EXPECT_EQ(ptr, nullptr);

  mem.deallocate(ptr);
}

TEST(cuda_um_memory, nullptr_deallocation_is_safe)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_um_memory<> mem("NULL_UM");

  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

TEST(cuda_um_memory, allocation_failure_throws_exception)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_um_memory<> mem("FAIL_UM");

  std::size_t huge_size = std::numeric_limits<std::size_t>::max() - 1024;
  EXPECT_THROW(mem.allocate(huge_size), umpire::out_of_memory_error);
}

TEST(cuda_um_memory, platform_type_is_cuda)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_um_memory<> mem("PLATFORM_UM");
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::cuda);
}

TEST(cuda_um_memory, convenience_aliases)
{
  static_assert(default_cuda_um_memory::tracking_enabled == true,
                "default_cuda_um_memory should have tracking enabled");
  static_assert(fast_cuda_um_memory::tracking_enabled == false,
                "fast_cuda_um_memory should have tracking disabled");
}

TEST(cuda_um_memory, type_traits)
{
  using tracked_um = cuda_um_memory<cuda_um_allocator, true>;
  using untracked_um = cuda_um_memory<cuda_um_allocator, false>;

  static_assert(std::is_same_v<tracked_um::platform, umpire::cuda_platform>,
                "Platform should be cuda_platform");
  static_assert(std::is_same_v<tracked_um::allocator_type, cuda_um_allocator>,
                "Allocator type should be cuda_um_allocator");
  static_assert(tracked_um::tracking_enabled == true, "tracking_enabled should be true");
  static_assert(untracked_um::tracking_enabled == false, "tracking_enabled should be false");
}

TEST(cuda_um_memory, deallocate_is_noexcept)
{
  using cuda_um = cuda_um_memory<>;
  static_assert(noexcept(std::declval<cuda_um>().deallocate(nullptr)),
                "deallocate() must be noexcept");
}

#endif // UMPIRE_ENABLE_CUDA

#if defined(UMPIRE_ENABLE_HIP)

#include "umpire/resource/hip_um_memory.hpp"

#include <hip/hip_runtime.h>

namespace {
bool hip_available() {
  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);
  return (error == hipSuccess && device_count > 0);
}
} // namespace

using namespace umpire::resource;

TEST(hip_um_memory, singleton_returns_same_instance)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  auto& inst1 = hip_um_memory<>::get();
  auto& inst2 = hip_um_memory<>::get();

  EXPECT_EQ(&inst1, &inst2);
  EXPECT_EQ(inst1.get_name(), "UM");
}

TEST(hip_um_memory, basic_allocation)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_um_memory<> mem("TEST_UM");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  char* cptr = static_cast<char*>(ptr);
  cptr[0] = 'A';
  cptr[1023] = 'Z';

  EXPECT_EQ(cptr[0], 'A');
  EXPECT_EQ(cptr[1023], 'Z');

  mem.deallocate(ptr);
}

TEST(hip_um_memory, tracking_enabled_records_allocation)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_um_memory<hip_um_allocator, true> mem("TRACKED_UM");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 1024u);
  EXPECT_EQ(mem.get_highwatermark(), 1024u);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0u);
}

TEST(hip_um_memory, zero_size_allocation_returns_nullptr)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_um_memory<> mem("ZERO_UM");

  void* ptr = mem.allocate(0);
  EXPECT_EQ(ptr, nullptr);
  mem.deallocate(ptr);
}

TEST(hip_um_memory, nullptr_deallocation_is_safe)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_um_memory<> mem("NULL_UM");
  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

TEST(hip_um_memory, platform_type_is_hip)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_um_memory<> mem("PLATFORM_UM");
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::hip);
}

TEST(hip_um_memory, type_traits)
{
  using tracked_um = hip_um_memory<hip_um_allocator, true>;
  using untracked_um = hip_um_memory<hip_um_allocator, false>;

  static_assert(std::is_same_v<tracked_um::platform, umpire::hip_platform>,
                "Platform should be hip_platform");
  static_assert(std::is_same_v<tracked_um::allocator_type, hip_um_allocator>,
                "Allocator type should be hip_um_allocator");
  static_assert(tracked_um::tracking_enabled == true, "tracking_enabled should be true");
  static_assert(untracked_um::tracking_enabled == false, "tracking_enabled should be false");
}

TEST(hip_um_memory, deallocate_is_noexcept)
{
  using hip_um = hip_um_memory<>;
  static_assert(noexcept(std::declval<hip_um>().deallocate(nullptr)),
                "deallocate() must be noexcept");
}

#endif // UMPIRE_ENABLE_HIP

#if defined(UMPIRE_ENABLE_SYCL)

#include "umpire/resource/sycl_um_memory.hpp"

#include <array>
#include <vector>

namespace {

bool supports_shared_allocations(const sycl::device& device)
{
  return device.has(sycl::aspect::usm_shared_allocations);
}

std::vector<sycl::device> sycl_supported_devices()
{
  std::vector<sycl::device> supported;

  try {
    for (const auto& device : sycl::device::get_devices()) {
      if (supports_shared_allocations(device)) {
        supported.push_back(device);
      }
    }
  } catch (...) {
    return {};
  }

  return supported;
}

bool sycl_device_available()
{
  return !sycl_supported_devices().empty();
}

sycl::device pick_test_device()
{
  const auto devices = sycl_supported_devices();

  for (const auto& device : devices) {
    if (device.is_gpu()) {
      return device;
    }
  }

  for (const auto& device : devices) {
    if (device.is_cpu()) {
      return device;
    }
  }

  return devices.front();
}

sycl::queue make_test_queue()
{
  return sycl::queue{pick_test_device()};
}

} // namespace

using namespace umpire::resource;

TEST(sycl_um_memory, custom_instance_with_queue)
{
  if (!sycl_device_available()) GTEST_SKIP() << "No SYCL device with USM shared allocations available";

  auto queue = make_test_queue();
  sycl_um_memory<> mem("SYCL_UM", queue);

  EXPECT_EQ(mem.get_name(), "SYCL_UM");
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::sycl);
  EXPECT_EQ(mem.get_queue().get_device(), queue.get_device());
}

TEST(sycl_um_memory, basic_allocation)
{
  if (!sycl_device_available()) GTEST_SKIP() << "No SYCL device with USM shared allocations available";

  auto queue = make_test_queue();
  sycl_um_memory<> mem("TEST_UM", queue);

  auto* ptr = static_cast<int*>(mem.allocate(4 * sizeof(int)));
  ASSERT_NE(ptr, nullptr);

  // Shared USM is directly host-accessible
  ptr[0] = 2;
  ptr[3] = 8;

  EXPECT_EQ(ptr[0], 2);
  EXPECT_EQ(ptr[3], 8);

  mem.deallocate(ptr);
}

TEST(sycl_um_memory, tracking_enabled_records_allocation)
{
  if (!sycl_device_available()) GTEST_SKIP() << "No SYCL device with USM shared allocations available";

  sycl_um_memory<sycl_um_allocator, true> mem("TRACKED_UM", make_test_queue());

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 1024u);
  EXPECT_EQ(mem.get_highwatermark(), 1024u);

  mem.deallocate(ptr);

  EXPECT_EQ(mem.get_current_size(), 0u);
  EXPECT_EQ(mem.get_highwatermark(), 1024u);
}

TEST(sycl_um_memory, zero_size_allocation_returns_nullptr)
{
  if (!sycl_device_available()) GTEST_SKIP() << "No SYCL device with USM shared allocations available";

  sycl_um_memory<> mem("ZERO_UM", make_test_queue());

  void* ptr = mem.allocate(0);
  EXPECT_EQ(ptr, nullptr);
  mem.deallocate(ptr);
}

TEST(sycl_um_memory, nullptr_deallocation_is_safe)
{
  if (!sycl_device_available()) GTEST_SKIP() << "No SYCL device with USM shared allocations available";

  sycl_um_memory<> mem("NULL_UM", make_test_queue());

  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

TEST(sycl_um_memory, type_traits)
{
  using tracked_um = sycl_um_memory<sycl_um_allocator, true>;
  using untracked_um = sycl_um_memory<sycl_um_allocator, false>;

  static_assert(std::is_same_v<tracked_um::platform, umpire::sycl_platform>,
                "Platform should be sycl_platform");
  static_assert(std::is_same_v<tracked_um::allocator_type, sycl_um_allocator>,
                "Allocator type should be sycl_um_allocator");
  static_assert(tracked_um::tracking_enabled == true, "tracking_enabled should be true");
  static_assert(untracked_um::tracking_enabled == false, "tracking_enabled should be false");
}

#endif // UMPIRE_ENABLE_SYCL

// Keep this translation unit non-empty even when no GPU backend is enabled.
#if !defined(UMPIRE_ENABLE_CUDA) && !defined(UMPIRE_ENABLE_HIP) && !defined(UMPIRE_ENABLE_SYCL)
TEST(um_memory, no_backend_enabled)
{
  GTEST_SKIP() << "No UM-capable backend (CUDA/HIP/SYCL) enabled in this build";
}
#endif
