#include "gtest/gtest.h"

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"

// #include "forall.hpp"

#define ALLOCATE(x) \
   void* allocations[x]; \
   for (int i = 0; i < x; i++) { \
     allocations[i] = allocator.allocate(i+1); \
   }

#define DEALLOCATE(x) \
  for (int i = 0; i < x; i++) { \
    allocator.deallocate(allocations[i+1]); \
  }

#define CUDA_TEST(X, Y) \
static void cuda_test_ ## X ## Y();\
TEST(X,Y) { cuda_test_ ## X ## Y();}\
static void cuda_test_ ## X ## Y()


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

  ALLOCATE(256);

  ASSERT_EQ(allocator.getSize(allocations[127]), 128);

  DEALLOCATE(256);
}

#if defined(ENABLE_CUDA)
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

  ALLOCATE(256);

  ASSERT_EQ(allocator.getSize(allocations[127]), 128);

  DEALLOCATE(256);
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

  ALLOCATE(256);

  ASSERT_EQ(allocator.getSize(allocations[127]), 128);

  DEALLOCATE(256);
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
  ASSERT_EQ(allocator.getName(), "host_monontonic_pool");

  ALLOCATE(256);

  ASSERT_EQ(allocator.getSize(allocations[127]), 128);

  DEALLOCATE(256);

}

#if defined(ENABLE_CUDA)
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

  ALLOCATE(256);

  ASSERT_EQ(allocator.getSize(allocations[127]), 128);

  DEALLOCATE(256);
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

  ALLOCATE(256);

  ASSERT_EQ(allocator.getSize(allocations[127]), 128);

  DEALLOCATE(256);
}
#endif
