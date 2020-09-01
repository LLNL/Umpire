//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/alloc/MallocAllocator.hpp"
#include "umpire/config.hpp"

using namespace umpire::alloc;

#if defined(UMPIRE_ENABLE_CUDA)
#include "umpire/alloc/CudaMallocAllocator.hpp"
#include "umpire/alloc/CudaMallocManagedAllocator.hpp"
#include "umpire/alloc/CudaPinnedAllocator.hpp"
#endif

#if defined(UMPIRE_ENABLE_HIP)
#include "umpire/alloc/HipMallocAllocator.hpp"
#include "umpire/alloc/HipPinnedAllocator.hpp"
#endif

#include "gtest/gtest.h"

template <typename T>
class MemoryAllocatorTest : public ::testing::Test {
};

TYPED_TEST_SUITE_P(MemoryAllocatorTest);

TYPED_TEST_P(MemoryAllocatorTest, Allocate)
{
  TypeParam allocator;
  void* allocation = allocator.allocate(1000);
  ASSERT_NE(nullptr, allocation);

  allocator.deallocate(allocation);
}

REGISTER_TYPED_TEST_SUITE_P(MemoryAllocatorTest, Allocate);

using test_types =
    ::testing::Types<MallocAllocator
#if defined(UMPIRE_ENABLE_CUDA)
                     ,
                     CudaMallocAllocator, CudaMallocManagedAllocator,
                     CudaPinnedAllocator
#endif
#if defined(UMPIRE_ENABLE_HIP)
                     ,
                     HipMallocAllocator, HipPinnedAllocator
#endif
                     >;

INSTANTIATE_TYPED_TEST_SUITE_P(Default, MemoryAllocatorTest, test_types, );
