#include "gtest/gtest.h"

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"

TEST(Allocator, HostAllocator)
{
  auto rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("HOST");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);
}

#if defined(ENABLE_CUDA)
TEST(Allocator, DeviceAllocator)
{
  auto rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("DEVICE");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);
}

TEST(Allocator, UmAllocator)
{
  auto rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("UM");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);
}
#endif
