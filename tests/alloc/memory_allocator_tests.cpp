#include "CudaMallocAllocator.hpp"
#include "MallocAllocator.hpp"
#include "PosixMemalignAllocator.hpp"

#include "gtest/gtest.h"

template <typename T>
class MemoryAllocatorTest : public ::testing::Test {
};

TYPED_TEST_CASE_P(MemoryAllocatorTest);

TYPED_TEST_P(MemoryAllocatorTest, Malloc) {
  TypeParam allocator;
  allocator.malloc();
}

TYPED_TEST_P(MemoryAllocatorTest, Calloc) {
}

TYPED_TEST_P(MemoryAllocatorTest, Realloc) {
}

TYPED_TEST_P(MemoryAllocatorTest, Free) {
}

TEST(Allocator, ResourceManager) {
  umpire::ResourceManager rm = umpire::ResourceManager::getInstance();
  ASSERT_NE(test, nullptr);
}
