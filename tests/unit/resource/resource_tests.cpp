#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "umpire/resource/DefaultMemoryResource.hpp"

struct TestAllocator
{
  void* allocate(size_t bytes) 
  {
    return ::malloc(bytes);
  }

  void deallocate(void* ptr)
  {
    ::free(ptr);
  }
};

TEST(DefaultMemoryResource, Constructor)
{
  auto alloc = std::make_shared<umpire::resource::DefaultMemoryResource<TestAllocator> >(umpire::Platform::cpu, "TEST");

  SUCCEED();
}

TEST(DefaultMemoryResource, AllocateDeallocate)
{
  auto alloc = std::make_shared<umpire::resource::DefaultMemoryResource<TestAllocator> >(umpire::Platform::cpu, "TEST");
  double* pointer = (double*)alloc->allocate(10*sizeof(double));
  ASSERT_NE(pointer, nullptr);
}

TEST(DefaultMemoryResource, GetSize)
{
  auto alloc = std::make_shared<umpire::resource::DefaultMemoryResource<TestAllocator> >(umpire::Platform::cpu, "TEST");
  double* pointer = (double*) alloc->allocate(10);
  ASSERT_EQ(alloc->getCurrentSize(), 10);

  double* pointer_two = (double*)alloc->allocate(10);
  ASSERT_EQ(alloc->getCurrentSize(), 20);

  alloc->deallocate(pointer);
  ASSERT_EQ(alloc->getCurrentSize(), 10);

  alloc->deallocate(pointer_two);
  ASSERT_EQ(alloc->getCurrentSize(), 0);
}

TEST(DefaultMemoryResource, GetHighWatermark)
{
  auto alloc = std::make_shared<umpire::resource::DefaultMemoryResource<TestAllocator> >(umpire::Platform::cpu, "TEST");
  ASSERT_EQ(alloc->getHighWatermark(), 0);

  double* pointer = (double*)alloc->allocate(10);
  double* pointer_two = (double*)alloc->allocate(30);

  alloc->deallocate(pointer);

  ASSERT_EQ(alloc->getHighWatermark(), 40);

  alloc->deallocate(pointer_two);
}
