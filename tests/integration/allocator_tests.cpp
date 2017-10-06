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

TEST(Allocator, HostAllocatorReference)
{
  auto rm = umpire::ResourceManager::getInstance();
  umpire::Allocator *p;

  p = new umpire::Allocator(rm.getAllocator("HOST"));

  double* test_alloc = static_cast<double*>(p->allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);

  p->deallocate(test_alloc);

  delete p;
}

TEST(Allocator, HostAllocatorSize)
{
  auto rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("HOST");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_EQ((100*sizeof(double)), allocator.size(test_alloc));

  allocator.deallocate(test_alloc);

  ASSERT_ANY_THROW(allocator.size(test_alloc));
}

#if defined(ENABLE_CUDA)
TEST(Allocator, DeviceAllocator)
{
  auto rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("DEVICE");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);
}

TEST(Allocator, DeviceAllocatorReference)
{
  auto rm = umpire::ResourceManager::getInstance();
  umpire::Allocator *p;

  p = new umpire::Allocator(rm.getAllocator("DEVICE"));

  double* test_alloc = static_cast<double*>(p->allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);

  p->deallocate(test_alloc);

  delete p;
}

TEST(Allocator, DeviceAllocatorSize)
{
  auto rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("DEVICE");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_EQ((100*sizeof(double)), allocator.size(test_alloc));

  allocator.deallocate(test_alloc);

  ASSERT_ANY_THROW(allocator.size(test_alloc));
}

TEST(Allocator, UmAllocator)
{
  auto rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("UM");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);
}

TEST(Allocator, UmAllocatorReference)
{
  auto rm = umpire::ResourceManager::getInstance();
  umpire::Allocator *p;

  p = new umpire::Allocator(rm.getAllocator("UM"));

  double* test_alloc = static_cast<double*>(p->allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);

  p->deallocate(test_alloc);

  delete p;
}

TEST(Allocator, UmAllocatorSize)
{
  auto rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("UM");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_EQ((100*sizeof(double)), allocator.size(test_alloc));

  allocator.deallocate(test_alloc);

  ASSERT_ANY_THROW(allocator.size(test_alloc));
}

#endif

TEST(Allocator, Deallocate)
{
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("HOST");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  rm.deallocate(test_alloc);

  SUCCEED();
}

TEST(Allocator, DeallocateThrow)
{
  auto& rm = umpire::ResourceManager::getInstance();

  double* ptr = new double[20];
  ASSERT_ANY_THROW(rm.deallocate(ptr));
}
