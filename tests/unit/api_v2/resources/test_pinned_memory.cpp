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

#include "umpire/resource/cuda_pinned_memory.hpp"

#include <cuda_runtime_api.h>

namespace {
bool cuda_available() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  return (error == cudaSuccess && device_count > 0);
}
} // namespace

using namespace umpire::resource;

TEST(cuda_pinned_memory, singleton_returns_same_instance)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  auto& inst1 = cuda_pinned_memory<>::get();
  auto& inst2 = cuda_pinned_memory<>::get();

  EXPECT_EQ(&inst1, &inst2);
  EXPECT_EQ(inst1.get_name(), "PINNED");
}

TEST(cuda_pinned_memory, custom_instance_with_custom_name)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_pinned_memory<> custom("CUSTOM_PINNED");
  EXPECT_EQ(custom.get_name(), "CUSTOM_PINNED");
}

TEST(cuda_pinned_memory, basic_allocation)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_pinned_memory<> mem("TEST_PINNED");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  // Pinned host memory is directly host-accessible
  char* cptr = static_cast<char*>(ptr);
  cptr[0] = 'A';
  cptr[1023] = 'Z';

  EXPECT_EQ(cptr[0], 'A');
  EXPECT_EQ(cptr[1023], 'Z');

  mem.deallocate(ptr);
}

TEST(cuda_pinned_memory, tracking_enabled_records_allocation)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_pinned_memory<cuda_pinned_allocator, true> mem("TRACKED_PINNED");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 1024u);
  EXPECT_EQ(mem.get_highwatermark(), 1024u);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0u);
  EXPECT_EQ(mem.get_highwatermark(), 1024u);
}

TEST(cuda_pinned_memory, tracking_disabled_no_registry_interaction)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_pinned_memory<cuda_pinned_allocator, false> mem("UNTRACKED_PINNED");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 0u);
  EXPECT_EQ(mem.get_highwatermark(), 0u);

  mem.deallocate(ptr);
}

TEST(cuda_pinned_memory, zero_size_allocation_returns_nullptr)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_pinned_memory<> mem("ZERO_PINNED");

  void* ptr = mem.allocate(0);
  EXPECT_EQ(ptr, nullptr);
  mem.deallocate(ptr);
}

TEST(cuda_pinned_memory, nullptr_deallocation_is_safe)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_pinned_memory<> mem("NULL_PINNED");
  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

TEST(cuda_pinned_memory, allocation_failure_throws_exception)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_pinned_memory<> mem("FAIL_PINNED");

  std::size_t huge_size = std::numeric_limits<std::size_t>::max() - 1024;
  EXPECT_THROW(mem.allocate(huge_size), umpire::out_of_memory_error);
}

TEST(cuda_pinned_memory, platform_type_is_cuda)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_pinned_memory<> mem("PLATFORM_PINNED");
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::cuda);
}

TEST(cuda_pinned_memory, convenience_aliases)
{
  static_assert(default_cuda_pinned_memory::tracking_enabled == true,
                "default_cuda_pinned_memory should have tracking enabled");
  static_assert(fast_cuda_pinned_memory::tracking_enabled == false,
                "fast_cuda_pinned_memory should have tracking disabled");
}

TEST(cuda_pinned_memory, type_traits)
{
  using tracked_pinned = cuda_pinned_memory<cuda_pinned_allocator, true>;
  using untracked_pinned = cuda_pinned_memory<cuda_pinned_allocator, false>;

  static_assert(std::is_same_v<tracked_pinned::platform, umpire::cuda_platform>,
                "Platform should be cuda_platform");
  static_assert(std::is_same_v<tracked_pinned::allocator_type, cuda_pinned_allocator>,
                "Allocator type should be cuda_pinned_allocator");
  static_assert(tracked_pinned::tracking_enabled == true, "tracking_enabled should be true");
  static_assert(untracked_pinned::tracking_enabled == false, "tracking_enabled should be false");
}

TEST(cuda_pinned_memory, deallocate_is_noexcept)
{
  using cuda_pinned = cuda_pinned_memory<>;
  static_assert(noexcept(std::declval<cuda_pinned>().deallocate(nullptr)),
                "deallocate() must be noexcept");
}

#endif // UMPIRE_ENABLE_CUDA

#if defined(UMPIRE_ENABLE_HIP)

#include "umpire/resource/hip_pinned_memory.hpp"

#include <hip/hip_runtime.h>

namespace {
bool hip_available() {
  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);
  return (error == hipSuccess && device_count > 0);
}
} // namespace

using namespace umpire::resource;

TEST(hip_pinned_memory, singleton_returns_same_instance)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  auto& inst1 = hip_pinned_memory<>::get();
  auto& inst2 = hip_pinned_memory<>::get();

  EXPECT_EQ(&inst1, &inst2);
  EXPECT_EQ(inst1.get_name(), "PINNED");
}

TEST(hip_pinned_memory, basic_allocation)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_pinned_memory<> mem("TEST_PINNED");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  char* cptr = static_cast<char*>(ptr);
  cptr[0] = 'A';
  cptr[1023] = 'Z';

  EXPECT_EQ(cptr[0], 'A');
  EXPECT_EQ(cptr[1023], 'Z');

  mem.deallocate(ptr);
}

TEST(hip_pinned_memory, tracking_enabled_records_allocation)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_pinned_memory<hip_pinned_allocator, true> mem("TRACKED_PINNED");

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 1024u);
  EXPECT_EQ(mem.get_highwatermark(), 1024u);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0u);
}

TEST(hip_pinned_memory, zero_size_allocation_returns_nullptr)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_pinned_memory<> mem("ZERO_PINNED");

  void* ptr = mem.allocate(0);
  EXPECT_EQ(ptr, nullptr);
  mem.deallocate(ptr);
}

TEST(hip_pinned_memory, nullptr_deallocation_is_safe)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_pinned_memory<> mem("NULL_PINNED");
  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

TEST(hip_pinned_memory, platform_type_is_hip)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_pinned_memory<> mem("PLATFORM_PINNED");
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::hip);
}

TEST(hip_pinned_memory, type_traits)
{
  using tracked_pinned = hip_pinned_memory<hip_pinned_allocator, true>;
  using untracked_pinned = hip_pinned_memory<hip_pinned_allocator, false>;

  static_assert(std::is_same_v<tracked_pinned::platform, umpire::hip_platform>,
                "Platform should be hip_platform");
  static_assert(std::is_same_v<tracked_pinned::allocator_type, hip_pinned_allocator>,
                "Allocator type should be hip_pinned_allocator");
  static_assert(tracked_pinned::tracking_enabled == true, "tracking_enabled should be true");
  static_assert(untracked_pinned::tracking_enabled == false, "tracking_enabled should be false");
}

TEST(hip_pinned_memory, deallocate_is_noexcept)
{
  using hip_pinned = hip_pinned_memory<>;
  static_assert(noexcept(std::declval<hip_pinned>().deallocate(nullptr)),
                "deallocate() must be noexcept");
}

#endif // UMPIRE_ENABLE_HIP

#if defined(UMPIRE_ENABLE_SYCL)

#include "umpire/resource/sycl_pinned_memory.hpp"

#include <array>
#include <vector>

namespace {

bool supports_host_allocations(const sycl::device& device)
{
  return device.has(sycl::aspect::usm_host_allocations);
}

std::vector<sycl::device> sycl_supported_devices()
{
  std::vector<sycl::device> supported;

  try {
    for (const auto& device : sycl::device::get_devices()) {
      if (supports_host_allocations(device)) {
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

TEST(sycl_pinned_memory, custom_instance_with_queue)
{
  if (!sycl_device_available()) GTEST_SKIP() << "No SYCL device with USM host allocations available";

  auto queue = make_test_queue();
  sycl_pinned_memory<> mem("SYCL_PINNED", queue);

  EXPECT_EQ(mem.get_name(), "SYCL_PINNED");
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::sycl);
  EXPECT_EQ(mem.get_queue().get_device(), queue.get_device());
}

TEST(sycl_pinned_memory, basic_allocation)
{
  if (!sycl_device_available()) GTEST_SKIP() << "No SYCL device with USM host allocations available";

  auto queue = make_test_queue();
  sycl_pinned_memory<> mem("TEST_PINNED", queue);

  auto* ptr = static_cast<int*>(mem.allocate(4 * sizeof(int)));
  ASSERT_NE(ptr, nullptr);

  ptr[0] = 2;
  ptr[3] = 8;

  EXPECT_EQ(ptr[0], 2);
  EXPECT_EQ(ptr[3], 8);

  mem.deallocate(ptr);
}

TEST(sycl_pinned_memory, tracking_enabled_records_allocation)
{
  if (!sycl_device_available()) GTEST_SKIP() << "No SYCL device with USM host allocations available";

  sycl_pinned_memory<sycl_pinned_allocator, true> mem("TRACKED_PINNED", make_test_queue());

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(mem.get_current_size(), 1024u);
  EXPECT_EQ(mem.get_highwatermark(), 1024u);

  mem.deallocate(ptr);

  EXPECT_EQ(mem.get_current_size(), 0u);
  EXPECT_EQ(mem.get_highwatermark(), 1024u);
}

TEST(sycl_pinned_memory, zero_size_allocation_returns_nullptr)
{
  if (!sycl_device_available()) GTEST_SKIP() << "No SYCL device with USM host allocations available";

  sycl_pinned_memory<> mem("ZERO_PINNED", make_test_queue());

  void* ptr = mem.allocate(0);
  EXPECT_EQ(ptr, nullptr);
  mem.deallocate(ptr);
}

TEST(sycl_pinned_memory, nullptr_deallocation_is_safe)
{
  if (!sycl_device_available()) GTEST_SKIP() << "No SYCL device with USM host allocations available";

  sycl_pinned_memory<> mem("NULL_PINNED", make_test_queue());

  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

TEST(sycl_pinned_memory, type_traits)
{
  using tracked_pinned = sycl_pinned_memory<sycl_pinned_allocator, true>;
  using untracked_pinned = sycl_pinned_memory<sycl_pinned_allocator, false>;

  static_assert(std::is_same_v<tracked_pinned::platform, umpire::sycl_platform>,
                "Platform should be sycl_platform");
  static_assert(std::is_same_v<tracked_pinned::allocator_type, sycl_pinned_allocator>,
                "Allocator type should be sycl_pinned_allocator");
  static_assert(tracked_pinned::tracking_enabled == true, "tracking_enabled should be true");
  static_assert(untracked_pinned::tracking_enabled == false, "tracking_enabled should be false");
}

#endif // UMPIRE_ENABLE_SYCL

// Keep this translation unit non-empty even when no pinned-capable backend is enabled.
#if !defined(UMPIRE_ENABLE_CUDA) && !defined(UMPIRE_ENABLE_HIP) && !defined(UMPIRE_ENABLE_SYCL)
TEST(pinned_memory, no_backend_enabled)
{
  GTEST_SKIP() << "No pinned-capable backend (CUDA/HIP/SYCL) enabled in this build";
}
#endif
