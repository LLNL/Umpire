//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/config.hpp"

#if defined(UMPIRE_ENABLE_CUDA)

#include "umpire/resource/cuda_device_memory.hpp"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <type_traits>

using namespace umpire::resource;

// Helper function to check if CUDA is available
bool cuda_available() {
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  return (error == cudaSuccess && device_count > 0);
}

// Test that singleton returns the same instance each time
TEST(cuda_device_memory, singleton_returns_same_instance)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  auto& inst1 = cuda_device_memory<>::get();
  auto& inst2 = cuda_device_memory<>::get();

  // Should be the exact same instance
  EXPECT_EQ(&inst1, &inst2);
  EXPECT_EQ(inst1.get_name(), "CUDA");
  EXPECT_EQ(inst1.get_device_id(), 0);
}

// Test that custom instance can be constructed with custom name and device
TEST(cuda_device_memory, custom_instance_with_custom_name)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_memory<> custom("CUSTOM_CUDA", 0);
  EXPECT_EQ(custom.get_name(), "CUSTOM_CUDA");
  EXPECT_EQ(custom.get_device_id(), 0);
}

// Test custom instance with just device_id
TEST(cuda_device_memory, custom_instance_with_device_id)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_memory<> custom(0);
  EXPECT_EQ(custom.get_name(), "CUDA_0");
  EXPECT_EQ(custom.get_device_id(), 0);
}

// Test basic allocation: allocate(1024), verify non-null, deallocate
TEST(cuda_device_memory, basic_allocation)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_memory<> mem("TEST_CUDA", 0);

  void* ptr = mem.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  // Verify we can write to the memory from device
  // Copy pattern to device, read it back to verify
  std::vector<char> host_data(1024);
  host_data[0] = 'A';
  host_data[1023] = 'Z';

  cudaError_t error = cudaMemcpy(ptr, host_data.data(), 1024, cudaMemcpyHostToDevice);
  ASSERT_EQ(error, cudaSuccess);

  std::vector<char> readback(1024);
  error = cudaMemcpy(readback.data(), ptr, 1024, cudaMemcpyDeviceToHost);
  ASSERT_EQ(error, cudaSuccess);

  EXPECT_EQ(readback[0], 'A');
  EXPECT_EQ(readback[1023], 'Z');

  mem.deallocate(ptr);
}

// Test tracking enabled: allocate, verify registry has record, deallocate, verify removed
TEST(cuda_device_memory, tracking_enabled_records_allocation)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_memory<cuda_default_allocator, true> mem("TRACKED_CUDA", 0);

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
TEST(cuda_device_memory, tracking_disabled_no_registry_interaction)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_memory<cuda_default_allocator, false> mem("UNTRACKED_CUDA", 0);

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
TEST(cuda_device_memory, large_allocation)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_memory<> mem("LARGE_CUDA", 0);

  // Get device memory info to determine safe allocation size
  size_t free_mem, total_mem;
  cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);
  ASSERT_EQ(error, cudaSuccess);

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
  error = cudaMemcpy(ptr, &first, 1, cudaMemcpyHostToDevice);
  ASSERT_EQ(error, cudaSuccess);
  error = cudaMemcpy(static_cast<char*>(ptr) + size - 1, &last, 1, cudaMemcpyHostToDevice);
  ASSERT_EQ(error, cudaSuccess);

  char readback_first, readback_last;
  error = cudaMemcpy(&readback_first, ptr, 1, cudaMemcpyDeviceToHost);
  ASSERT_EQ(error, cudaSuccess);
  error = cudaMemcpy(&readback_last, static_cast<char*>(ptr) + size - 1, 1, cudaMemcpyDeviceToHost);
  ASSERT_EQ(error, cudaSuccess);

  EXPECT_EQ(readback_first, 'A');
  EXPECT_EQ(readback_last, 'Z');

  mem.deallocate(ptr);
}

// Test zero-size allocation returns nullptr
TEST(cuda_device_memory, zero_size_allocation_returns_nullptr)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_memory<> mem("ZERO_CUDA", 0);

  void* ptr = mem.allocate(0);
  EXPECT_EQ(ptr, nullptr);

  // Deallocate should be safe no-op
  mem.deallocate(ptr);
}

// Test nullptr deallocation is safe no-op
TEST(cuda_device_memory, nullptr_deallocation_is_safe)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_memory<> mem("NULL_CUDA", 0);

  // Should not crash or throw
  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

// Test multiple allocations with correct statistics
TEST(cuda_device_memory, multiple_allocations_correct_statistics)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_memory<> mem("MULTI_CUDA", 0);

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
TEST(cuda_device_memory, allocation_failure_throws_exception)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_memory<> mem("FAIL_CUDA", 0);

  // Try to allocate an impossibly large amount
  // This should fail and throw out_of_memory_error
  std::size_t huge_size = std::numeric_limits<std::size_t>::max() - 1024;
  EXPECT_THROW(mem.allocate(huge_size), umpire::out_of_memory_error);
}

// Test platform type is correct
TEST(cuda_device_memory, platform_type_is_cuda)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_memory<> mem("PLATFORM_CUDA", 0);
  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::cuda);
}

// Test convenience aliases
TEST(cuda_device_memory, convenience_aliases)
{
  // default_cuda_device_memory should have tracking enabled
  static_assert(default_cuda_device_memory::tracking_enabled == true,
                "default_cuda_device_memory should have tracking enabled");

  // fast_cuda_device_memory should have tracking disabled
  static_assert(fast_cuda_device_memory::tracking_enabled == false,
                "fast_cuda_device_memory should have tracking disabled");
}

// Test type traits
TEST(cuda_device_memory, type_traits)
{
  using tracked_cuda = cuda_device_memory<cuda_default_allocator, true>;
  using untracked_cuda = cuda_device_memory<cuda_default_allocator, false>;

  // Platform type
  static_assert(std::is_same_v<tracked_cuda::platform, umpire::cuda_platform>,
                "Platform should be cuda_platform");

  // Allocator type
  static_assert(std::is_same_v<tracked_cuda::allocator_type, cuda_default_allocator>,
                "Allocator type should be cuda_default_allocator");

  // Tracking flag
  static_assert(tracked_cuda::tracking_enabled == true,
                "tracking_enabled should be true");
  static_assert(untracked_cuda::tracking_enabled == false,
                "tracking_enabled should be false");
}

// Test that allocations from different instances are independent
TEST(cuda_device_memory, independent_instances)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_memory<> mem1("CUDA1", 0);
  cuda_device_memory<> mem2("CUDA2", 0);

  void* ptr1 = mem1.allocate(100);
  void* ptr2 = mem2.allocate(200);

  EXPECT_EQ(mem1.get_current_size(), 100);
  EXPECT_EQ(mem2.get_current_size(), 200);

  mem1.deallocate(ptr1);
  mem2.deallocate(ptr2);
}

// Test many small allocations
TEST(cuda_device_memory, many_small_allocations)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  cuda_device_memory<> mem("MANY_SMALL", 0);

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
TEST(cuda_device_memory, multi_gpu_support)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  ASSERT_EQ(error, cudaSuccess);

  if (device_count < 2) {
    GTEST_SKIP() << "Multi-GPU test requires at least 2 CUDA devices";
  }

  // Create resources for different devices
  cuda_device_memory<> mem0("CUDA_DEV0", 0);
  cuda_device_memory<> mem1("CUDA_DEV1", 1);

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
TEST(cuda_device_memory, invalid_device_id_throws)
{
  if (!cuda_available()) GTEST_SKIP() << "CUDA not available";

  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);
  ASSERT_EQ(error, cudaSuccess);

  // Try to create resource with invalid device ID
  EXPECT_THROW(cuda_device_memory<>("INVALID", device_count), umpire::runtime_error);
  EXPECT_THROW(cuda_device_memory<>("INVALID", -1), umpire::runtime_error);
}

// Test deallocate is noexcept
TEST(cuda_device_memory, deallocate_is_noexcept)
{
  // This is a compile-time test
  using cuda_mem = cuda_device_memory<>;
  static_assert(noexcept(std::declval<cuda_mem>().deallocate(nullptr)),
                "deallocate() must be noexcept");
}

#endif // UMPIRE_ENABLE_CUDA
