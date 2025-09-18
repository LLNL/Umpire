//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <cstring>
#include <random>

#include "gtest/gtest.h"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/op.hpp"

// Host prefetch tests - these are mostly no-ops but ensure API compatibility
TEST(Prefetch, HostPrefetchNoOp)
{
  constexpr std::size_t size = 1024;

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  // Allocate host buffer
  void* ptr = allocator.allocate(size);

  // Fill with test data
  std::memset(ptr, 0x42, size);

  // Prefetch should be a no-op for host memory, but should not crash
  EXPECT_NO_THROW(umpire::prefetch(ptr, 0, size));

  // Data should be unchanged
  unsigned char* byte_ptr = static_cast<unsigned char*>(ptr);
  for (std::size_t i = 0; i < size; ++i) {
    ASSERT_EQ(byte_ptr[i], 0x42) << "Host prefetch modified data at byte " << i;
  }

  // Cleanup
  allocator.deallocate(ptr);
}

TEST(Prefetch, HostPrefetchExplicit)
{
  constexpr std::size_t size = 1024;

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  // Allocate host buffer
  void* ptr = allocator.allocate(size);

  // Fill with test data
  std::memset(ptr, 0x55, size);

  // Explicit host prefetch - should be a no-op
  EXPECT_NO_THROW(umpire::prefetch<umpire::host>(ptr, 0, size));

  // Data should be unchanged
  unsigned char* byte_ptr = static_cast<unsigned char*>(ptr);
  for (std::size_t i = 0; i < size; ++i) {
    ASSERT_EQ(byte_ptr[i], 0x55) << "Explicit host prefetch modified data at byte " << i;
  }

  // Cleanup
  allocator.deallocate(ptr);
}

TEST(Prefetch, ZeroSizePrefetch)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  // Allocate host buffer
  void* ptr = allocator.allocate(1024);

  // Zero-size prefetch should be safe
  EXPECT_NO_THROW(umpire::prefetch(ptr, 0, 0));

  // Cleanup
  allocator.deallocate(ptr);
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST(Prefetch, CudaPrefetchToHost)
{
  constexpr std::size_t size = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  // Check if we have managed memory support
  try {
    auto um_allocator = rm.getAllocator("UM");
    auto host_allocator = rm.getAllocator("HOST");

    // Allocate managed memory
    void* um_ptr = um_allocator.allocate(size);

    // Allocate host buffer for verification
    unsigned char* host_ptr = static_cast<unsigned char*>(host_allocator.allocate(size));

    // Fill managed memory with test data (on host)
    std::memset(um_ptr, 0x77, size);

    // Prefetch to host (CPU device ID is typically -1 or cudaCpuDeviceId)
    EXPECT_NO_THROW(umpire::prefetch(um_ptr, -1, size));

    // Copy to regular host memory to verify data integrity
    std::memcpy(host_ptr, um_ptr, size);

    // Verify data is intact after prefetch
    for (std::size_t i = 0; i < size; ++i) {
      ASSERT_EQ(host_ptr[i], 0x77) << "CUDA prefetch corrupted data at byte " << i;
    }

    // Cleanup
    um_allocator.deallocate(um_ptr);
    host_allocator.deallocate(host_ptr);
  } catch (const std::runtime_error& e) {
    // If UM allocator is not available, skip this test
    GTEST_SKIP() << "Unified Memory not available: " << e.what();
  }
}

TEST(Prefetch, CudaPrefetchToDevice)
{
  constexpr std::size_t size = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  // Check if we have managed memory support
  try {
    auto um_allocator = rm.getAllocator("UM");
    auto host_allocator = rm.getAllocator("HOST");

    // Allocate managed memory
    void* um_ptr = um_allocator.allocate(size);

    // Allocate host buffer for verification
    unsigned char* host_ptr = static_cast<unsigned char*>(host_allocator.allocate(size));

    // Fill managed memory with test data
    std::memset(um_ptr, 0x88, size);

    // Prefetch to device 0
    EXPECT_NO_THROW(umpire::prefetch(um_ptr, 0, size));

    // Copy to host memory to verify data integrity
    std::memcpy(host_ptr, um_ptr, size);

    // Verify data is intact after prefetch
    for (std::size_t i = 0; i < size; ++i) {
      ASSERT_EQ(host_ptr[i], 0x88) << "CUDA device prefetch corrupted data at byte " << i;
    }

    // Cleanup
    um_allocator.deallocate(um_ptr);
    host_allocator.deallocate(host_ptr);
  } catch (const std::runtime_error& e) {
    // If UM allocator is not available, skip this test
    GTEST_SKIP() << "Unified Memory not available: " << e.what();
  }
}

TEST(Prefetch, ExplicitCudaPrefetch)
{
  constexpr std::size_t size = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  // Check if we have managed memory support
  try {
    auto um_allocator = rm.getAllocator("UM");
    auto host_allocator = rm.getAllocator("HOST");

    // Allocate managed memory
    void* um_ptr = um_allocator.allocate(size);

    // Allocate host buffer for verification
    unsigned char* host_ptr = static_cast<unsigned char*>(host_allocator.allocate(size));

    // Fill managed memory with test data
    std::memset(um_ptr, 0x99, size);

    // Explicit CUDA prefetch to host
    EXPECT_NO_THROW(umpire::prefetch<umpire::cuda>(um_ptr, -1, size));

    // Copy to host memory to verify data integrity
    std::memcpy(host_ptr, um_ptr, size);

    // Verify data is intact after prefetch
    for (std::size_t i = 0; i < size; ++i) {
      ASSERT_EQ(host_ptr[i], 0x99) << "Explicit CUDA prefetch corrupted data at byte " << i;
    }

    // Cleanup
    um_allocator.deallocate(um_ptr);
    host_allocator.deallocate(host_ptr);
  } catch (const std::runtime_error& e) {
    // If UM allocator is not available, skip this test
    GTEST_SKIP() << "Unified Memory not available: " << e.what();
  }
}
#endif // UMPIRE_ENABLE_CUDA

#if defined(UMPIRE_ENABLE_HIP)
TEST(Prefetch, HipPrefetchToHost)
{
  constexpr std::size_t size = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  // Check if we have managed memory support
  try {
    auto um_allocator = rm.getAllocator("UM");
    auto host_allocator = rm.getAllocator("HOST");

    // Allocate managed memory
    void* um_ptr = um_allocator.allocate(size);

    // Allocate host buffer for verification
    unsigned char* host_ptr = static_cast<unsigned char*>(host_allocator.allocate(size));

    // Fill managed memory with test data (on host)
    std::memset(um_ptr, 0xAA, size);

    // Prefetch to host (CPU device ID is typically -1 or hipCpuDeviceId)
    EXPECT_NO_THROW(umpire::prefetch(um_ptr, -1, size));

    // Copy to regular host memory to verify data integrity
    std::memcpy(host_ptr, um_ptr, size);

    // Verify data is intact after prefetch
    for (std::size_t i = 0; i < size; ++i) {
      ASSERT_EQ(host_ptr[i], 0xAA) << "HIP prefetch corrupted data at byte " << i;
    }

    // Cleanup
    um_allocator.deallocate(um_ptr);
    host_allocator.deallocate(host_ptr);
  } catch (const std::runtime_error& e) {
    // If UM allocator is not available, skip this test
    GTEST_SKIP() << "Unified Memory not available: " << e.what();
  }
}

TEST(Prefetch, HipPrefetchToDevice)
{
  constexpr std::size_t size = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  // Check if we have managed memory support
  try {
    auto um_allocator = rm.getAllocator("UM");
    auto host_allocator = rm.getAllocator("HOST");

    // Allocate managed memory
    void* um_ptr = um_allocator.allocate(size);

    // Allocate host buffer for verification
    unsigned char* host_ptr = static_cast<unsigned char*>(host_allocator.allocate(size));

    // Fill managed memory with test data
    std::memset(um_ptr, 0xBB, size);

    // Prefetch to device 0
    EXPECT_NO_THROW(umpire::prefetch(um_ptr, 0, size));

    // Copy to host memory to verify data integrity
    std::memcpy(host_ptr, um_ptr, size);

    // Verify data is intact after prefetch
    for (std::size_t i = 0; i < size; ++i) {
      ASSERT_EQ(host_ptr[i], 0xBB) << "HIP device prefetch corrupted data at byte " << i;
    }

    // Cleanup
    um_allocator.deallocate(um_ptr);
    host_allocator.deallocate(host_ptr);
  } catch (const std::runtime_error& e) {
    // If UM allocator is not available, skip this test
    GTEST_SKIP() << "Unified Memory not available: " << e.what();
  }
}

TEST(Prefetch, ExplicitHipPrefetch)
{
  constexpr std::size_t size = 1024;

  auto& rm = umpire::ResourceManager::getInstance();

  // Check if we have managed memory support
  try {
    auto um_allocator = rm.getAllocator("UM");
    auto host_allocator = rm.getAllocator("HOST");

    // Allocate managed memory
    void* um_ptr = um_allocator.allocate(size);

    // Allocate host buffer for verification
    unsigned char* host_ptr = static_cast<unsigned char*>(host_allocator.allocate(size));

    // Fill managed memory with test data
    std::memset(um_ptr, 0xCC, size);

    // Explicit HIP prefetch to host
    EXPECT_NO_THROW(umpire::prefetch<umpire::hip>(um_ptr, -1, size));

    // Copy to host memory to verify data integrity
    std::memcpy(host_ptr, um_ptr, size);

    // Verify data is intact after prefetch
    for (std::size_t i = 0; i < size; ++i) {
      ASSERT_EQ(host_ptr[i], 0xCC) << "Explicit HIP prefetch corrupted data at byte " << i;
    }

    // Cleanup
    um_allocator.deallocate(um_ptr);
    host_allocator.deallocate(host_ptr);
  } catch (const std::runtime_error& e) {
    // If UM allocator is not available, skip this test
    GTEST_SKIP() << "Unified Memory not available: " << e.what();
  }
}
#endif // UMPIRE_ENABLE_HIP