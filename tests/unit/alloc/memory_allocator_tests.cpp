#include "umpire/config.hpp"

#include "umpire/alloc/MallocAllocator.hpp"

using namespace umpire::alloc;

#if defined(ENABLE_CUDA)
#include "umpire/alloc/CudaMallocAllocator.hpp"
#include "umpire/alloc/CudaMallocManagedAllocator.hpp"
#include "umpire/alloc/CnmemAllocator.hpp"
#endif

#include "gtest/gtest.h"

template <typename T>
class MemoryAllocatorTest : public ::testing::Test {
};

TYPED_TEST_CASE_P(MemoryAllocatorTest);

TYPED_TEST_P(MemoryAllocatorTest, Allocate) {
  TypeParam allocator;
  void* allocation = allocator.allocate(1000);

  ASSERT_NE(nullptr, allocation);

  allocator.deallocate(allocation);
}

REGISTER_TYPED_TEST_CASE_P(
    MemoryAllocatorTest,
    Allocate);

#if defined(ENABLE_CUDA)
using test_types = ::testing::Types<MallocAllocator, CudaMallocAllocator, CudaMallocManagedAllocator, CnmemAllocator>;
#else
using test_types = ::testing::Types<MallocAllocator>;
#endif

INSTANTIATE_TYPED_TEST_CASE_P(Default, MemoryAllocatorTest, test_types);
