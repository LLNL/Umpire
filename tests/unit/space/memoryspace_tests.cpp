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

TEST(MemorySpace, getCurrentSize)
{
  auto alloc = std::make_shared<umpire::space::MemorySpace<TestAllocator> >();
  ASSERT_EQ(alloc->getCurrentSize(), 0);


  double* pointer = (double*)alloc->allocate(10*sizeof(double));
  ASSERT_EQ(alloc->getCurrentSize(), 10*sizeof(double));

  alloc->deallocate(pointer);
  ASSERT_EQ(alloc->getCurrentSize(), 0);
}
