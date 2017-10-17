#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "umpire/space/MemorySpace.hpp"

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

TEST(MemorySpace, Constructor)
{
  auto alloc = std::make_shared<umpire::space::MemorySpace<TestAllocator> >();

  SUCCEED();
}

TEST(MemorySpace, AllocateDeallocate)
{
  auto alloc = std::make_shared<umpire::space::MemorySpace<TestAllocator> >();
  double* pointer = (double*)alloc->allocate(10*sizeof(double));
  ASSERT_NE(pointer, nullptr);
}

TEST(MemorySpace, GetSize)
{
  auto alloc = std::shared_ptr<umpire::space::MemorySpace<TestAllocator> >();
  double* pointer = (double*)alloc->allocate(10);
  ASSERT_EQ(alloc->getCurrentSize(), 10);

  double* pointer_two = (double*)alloc->allocate(10);
  ASSERT_EQ(alloc->getCurrentSize(), 20);

  alloc->deallocate(pointer);
  ASSERT_EQ(alloc->getCurrentSize(), 10);

  alloc->deallocate(pointer_two);
  ASSERT_EQ(alloc->getCurrentSize(), 0);
}

TEST(MemorySpace, GetHighWatermark)
{
  auto alloc = std::shared_ptr<umpire::space::MemorySpace<TestAllocator> >();
  double* pointer = (double*)alloc->allocate(10*sizeof(double));
  double* pointer_two = (double*)alloc->allocate(30*sizeof(double));
  alloc->deallocate(pointer);

  ASSERT_EQ(alloc->getHighWatermark(), 40);

  alloc->deallocate(pointer_two);
}
