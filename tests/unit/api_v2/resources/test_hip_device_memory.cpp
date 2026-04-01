//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_HIP)

#include "umpire/resource/hip_device_memory.hpp"

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>
#include <type_traits>

using namespace umpire::resource;

// Helper function to check if HIP is available
bool hip_available() {
  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);
  return (error == hipSuccess && device_count > 0);
}

// Test that singleton returns the same instance each time
TEST(hip_device_memory, singleton_returns_same_instance)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  auto& inst1 = hip_device_memory<>::get();
  auto& inst2 = hip_device_memory<>::get();

  // Should be the exact same instance
  EXPECT_EQ(&inst1, &inst2);
  EXPECT_EQ(inst1.get_name(), "HIP_DEVICE");
  EXPECT_EQ(inst1.get_device_id(), 0);
}

// Test that custom instance can be constructed with custom name and device
TEST(hip_device_memory, custom_instance_with_custom_name)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_memory<> custom("CUSTOM_HIP", 0);
  EXPECT_EQ(custom.get_name(), "CUSTOM_HIP");
  EXPECT_EQ(custom.get_device_id(), 0);
}

// Test custom instance with just device_id
TEST(hip_device_memory, custom_instance_with_device_id)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_memory<> custom(0);
  EXPECT_EQ(custom.get_name(), "HIP_DEVICE_0");
  EXPECT_EQ(custom.get_device_id(), 0);
}

// Test basic allocation: allocate(1024), verify non-null, deallocate
TEST(hip_device_memory, basic_allocation)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_memory<> mem("TEST_HIP", 0);

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  // Verify we can write to the memory from device
  // Copy pattern to device, read it back to verify
  std::vector<char> host_data(1024);
  host_data[0] = 'A';
  host_data[1023] = 'Z';

  hipError_t error = hipMemcpy(ptr, host_data.data(), 1024, hipMemcpyHostToDevice);
  ASSERT_EQ(error, hipSuccess);

  std::vector<char> readback(1024);
  error = hipMemcpy(readback.data(), ptr, 1024, hipMemcpyDeviceToHost);
  ASSERT_EQ(error, hipSuccess);

  EXPECT_EQ(readback[0], 'A');
  EXPECT_EQ(readback[1023], 'Z');

  mem.deallocate(ptr);
}

// Test tracking enabled: allocate, verify registry has record, deallocate, verify removed
TEST(hip_device_memory, tracking_enabled_records_allocation)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_memory<hip_default_allocator, true> mem("TRACKED_HIP", 0);

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  // Verify allocation is tracked
  EXPECT_EQ(mem.get_current_size(), 1024);
  EXPECT_EQ(mem.get_highwatermark(), 1024);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 1024);
}

// Test tracking disabled: verify no registry interaction
TEST(hip_device_memory, tracking_disabled_no_registry_interaction)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_memory<hip_default_allocator, false> mem("UNTRACKED_HIP", 0);

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  // With tracking disabled, statistics should remain at 0
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 0);

  mem.deallocate(ptr);
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 0);
}

// Test large allocation
TEST(hip_device_memory, large_allocation)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_memory<> mem("LARGE_HIP", 0);

  // Get device memory info to determine safe allocation size
  size_t free_mem, total_mem;
  hipError_t error = hipMemGetInfo(&free_mem, &total_mem);
  ASSERT_EQ(error, hipSuccess);

  // Allocate 10% of free memory (should be safe on most systems)
  std::size_t size = free_mem / 10;
  if (size > 100 * 1024 * 1024) {
    size = 100 * 1024 * 1024; // Cap at 100MB for test speed
  }

  void* ptr = mem.allocate(size);
  ASSERT_NE(ptr, nullptr);

  // Write to first and last bytes to verify it's accessible
  char first = 'A';
  char last = 'Z';
  error = hipMemcpy(ptr, &first, 1, hipMemcpyHostToDevice);
  ASSERT_EQ(error, hipSuccess);
  error = hipMemcpy(static_cast<char*>(ptr) + size - 1, &last, 1, hipMemcpyHostToDevice);
  ASSERT_EQ(error, hipSuccess);

  char readback_first, readback_last;
  error = hipMemcpy(&readback_first, ptr, 1, hipMemcpyDeviceToHost);
  ASSERT_EQ(error, hipSuccess);
  error = hipMemcpy(&readback_last, static_cast<char*>(ptr) + size - 1, 1, hipMemcpyDeviceToHost);
  ASSERT_EQ(error, hipSuccess);

  EXPECT_EQ(readback_first, 'A');
  EXPECT_EQ(readback_last, 'Z');

  mem.deallocate(ptr);
}

// Test zero-size allocation returns nullptr
TEST(hip_device_memory, zero_size_allocation_returns_nullptr)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_memory<> mem("ZERO_HIP", 0);

  void* ptr = mem.allocate(0);
  EXPECT_EQ(ptr, nullptr);

  // Deallocate should be safe no-op
  mem.deallocate(ptr);
}

// Test nullptr deallocation is safe no-op
TEST(hip_device_memory, nullptr_deallocation_is_safe)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_memory<> mem("NULL_HIP", 0);

  // Should not crash or throw
  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

// Test multiple allocations with correct statistics
TEST(hip_device_memory, multiple_allocations_correct_statistics)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_memory<> mem("MULTI_HIP", 0);

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

// Test allocation failure throws exception
TEST(hip_device_memory, allocation_failure_throws_exception)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_memory<> mem("FAIL_HIP", 0);

  // Try to allocate an impossibly large amount
  // This should fail and throw out_of_memory_error
  std::size_t huge_size = std::numeric_limits<std::size_t>::max() - 1024;
  EXPECT_THROW(mem.allocate(huge_size), umpire::out_of_memory_error);
}

// Test platform type is correct
TEST(hip_device_memory, platform_type_is_hip)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_memory<> mem("PLATFORM_HIP", 0);
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::hip);
}

// Test convenience aliases
TEST(hip_device_memory, convenience_aliases)
{
  // default_hip_device_memory should have tracking enabled
  static_assert(default_hip_device_memory::tracking_enabled == true,
                "default_hip_device_memory should have tracking enabled");

  // fast_hip_device_memory should have tracking disabled
  static_assert(fast_hip_device_memory::tracking_enabled == false,
                "fast_hip_device_memory should have tracking disabled");
}

// Test type traits
TEST(hip_device_memory, type_traits)
{
  using tracked_hip = hip_device_memory<hip_default_allocator, true>;
  using untracked_hip = hip_device_memory<hip_default_allocator, false>;

  // Platform type
  static_assert(std::is_same_v<tracked_hip::platform, umpire::hip_platform>,
                "Platform should be hip_platform");

  // Allocator type
  static_assert(std::is_same_v<tracked_hip::allocator_type, hip_default_allocator>,
                "Allocator type should be hip_default_allocator");

  // Tracking flag
  static_assert(tracked_hip::tracking_enabled == true,
                "tracking_enabled should be true");
  static_assert(untracked_hip::tracking_enabled == false,
                "tracking_enabled should be false");
}

// Test that allocations from different instances are independent
TEST(hip_device_memory, independent_instances)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_memory<> mem1("HIP1", 0);
  hip_device_memory<> mem2("HIP2", 0);

  void* ptr1 = mem1.allocate(100);
  void* ptr2 = mem2.allocate(200);

  EXPECT_EQ(mem1.get_current_size(), 100);
  EXPECT_EQ(mem2.get_current_size(), 200);

  mem1.deallocate(ptr1);
  mem2.deallocate(ptr2);
}

// Test many small allocations
TEST(hip_device_memory, many_small_allocations)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  hip_device_memory<> mem("MANY_SMALL", 0);

  const int num_allocs = 1000;
  void* ptrs[num_allocs];

  // Allocate many small blocks
  for (int i = 0; i < num_allocs; ++i) {
    ptrs[i] = mem.allocate(16);
    ASSERT_NE(ptrs[i], nullptr);
  }

  EXPECT_EQ(mem.get_current_size(), num_allocs * 16);

  // Deallocate all
  for (int i = 0; i < num_allocs; ++i) {
    mem.deallocate(ptrs[i]);
  }

  EXPECT_EQ(mem.get_current_size(), 0);
}

// Test multi-GPU support (if multiple devices available)
TEST(hip_device_memory, multi_gpu_support)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);
  ASSERT_EQ(error, hipSuccess);

  if (device_count < 2) {
    GTEST_SKIP() << "Multi-GPU test requires at least 2 HIP devices";
  }

  // Create resources for different devices
  hip_device_memory<> mem0("HIP_DEV0", 0);
  hip_device_memory<> mem1("HIP_DEV1", 1);

  EXPECT_EQ(mem0.get_device_id(), 0);
  EXPECT_EQ(mem1.get_device_id(), 1);

  // Allocate on both devices
  void* ptr0 = mem0.allocate(1024);
  void* ptr1 = mem1.allocate(1024);

  ASSERT_NE(ptr0, nullptr);
  ASSERT_NE(ptr1, nullptr);

  // Verify independent tracking
  EXPECT_EQ(mem0.get_current_size(), 1024);
  EXPECT_EQ(mem1.get_current_size(), 1024);

  mem0.deallocate(ptr0);
  mem1.deallocate(ptr1);

  EXPECT_EQ(mem0.get_current_size(), 0);
  EXPECT_EQ(mem1.get_current_size(), 0);
}

// Test invalid device ID throws exception
TEST(hip_device_memory, invalid_device_id_throws)
{
  if (!hip_available()) GTEST_SKIP() << "HIP not available";

  int device_count = 0;
  hipError_t error = hipGetDeviceCount(&device_count);
  ASSERT_EQ(error, hipSuccess);

  // Try to create resource with invalid device ID
  EXPECT_THROW(hip_device_memory<>("INVALID", device_count), umpire::runtime_error);
  EXPECT_THROW(hip_device_memory<>("INVALID", -1), umpire::runtime_error);
}

// Test deallocate is noexcept
TEST(hip_device_memory, deallocate_is_noexcept)
{
  // This is a compile-time test
  using hip_mem = hip_device_memory<>;
  static_assert(noexcept(std::declval<hip_mem>().deallocate(nullptr)),
                "deallocate() must be noexcept");
}

#endif // UMPIRE_ENABLE_HIP
