#include "gtest/gtest.h"

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"

TEST(SimpoolStrategy, Host)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator(
      "host_simpool", "POOL", {}, {rm.getAllocator("HOST")});

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "host_simpool");
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST(SimpoolStrategy, Device)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator(
      "device_simpool", "POOL", {}, {rm.getAllocator("DEVICE")});

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "device_simpool");
}

TEST(SimpoolStrategy, UM)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator(
      "um_simpool", "POOL", {}, {rm.getAllocator("UM")});

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "um_simpool");
}
#endif

TEST(MonotonicStrategy, Host)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator(
      "host_monotonic_pool", "MONOTONIC", {65536}, {rm.getAllocator("HOST")});

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "host_monotonic_pool");
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST(MonotonicStrategy, Device)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator(
      "device_monotonic_pool", "MONOTONIC", {65536}, {rm.getAllocator("DEVICE")});

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "device_monotonic_pool");
}

TEST(MonotonicStrategy, UM)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator(
      "um_monotonic_pool", "MONOTONIC", {65536}, {rm.getAllocator("UM")});

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "um_monotonic_pool");
}
#endif
