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
  std::shared_ptr<umpire::space::MemorySpace<TestAllocator> > alloc;

  SUCCEED();
}

TEST(MemorySpace, AllocateDeallocate)
{
  std::shared_ptr<umpire::space::MemorySpace<TestAllocator> > alloc;
  void* pointer = alloc->allocate(10);
  ASSERT_NE(pointer, nullptr);
}

TEST(MemorySpace, GetSize)
{
  std::shared_ptr<umpire::space::MemorySpace<TestAllocator> > alloc;
  void* pointer = alloc->allocate(10);
  ASSERT_EQ(alloc->getCurrentSize(), 10);

  void* pointer_two = alloc->allocate(10);
  ASSERT_EQ(alloc->getCurrentSize(), 20);

  alloc->deallocate(pointer);
  ASSERT_EQ(alloc->getCurrentSize(), 10);

  alloc->deallocate(pointer_two);
  ASSERT_EQ(alloc->getCurrentSize(), 0);
}

TEST(MemorySpace, GetHighWatermark)
{
  std::shared_ptr<umpire::space::MemorySpace<TestAllocator> > alloc;
  void* pointer = alloc->allocate(10);
  void* pointer_two = alloc->allocate(30);
  alloc->deallocate(pointer);

  ASSERT_EQ(alloc->getHighWatermark(), 40);

  alloc->deallocate(pointer_two);
}
