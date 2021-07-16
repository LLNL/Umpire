//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/config.hpp"

#include "umpire/alloc/malloc_allocator.hpp"

#if defined(UMPIRE_ENABLE_CUDA)
#include "umpire/alloc/cuda_malloc_allocator.hpp"
#include "umpire/alloc/cuda_malloc_managed_allocator.hpp"
#include "umpire/alloc/cuda_pinned_allocator.hpp"
#endif

#if defined(UMPIRE_ENABLE_HIP)
#include "umpire/alloc/hip_malloc_allocator.hpp"
#include "umpire/alloc/hip_pinned_allocator.hpp"
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
#include "umpire/alloc/omp_target_allocator.hpp"
#endif

#include "gtest/gtest.h"

template <typename T>
class memory_allocator_test : public ::testing::Test {
};

TYPED_TEST_SUITE_P(memory_allocator_test);

TYPED_TEST_P(memory_allocator_test, allocate) {
  void* ptr = TypeParam::allocate(1000);
  EXPECT_NE(nullptr, allocation);
  TypeParam::deallocate(ptr);
}

REGISTER_TYPED_TEST_SUITE_P(
    memory_allocator_test,
    allocate);

using test_types = ::testing::Types<
    umpire::alloc::malloc_allocator
#if defined(UMPIRE_ENABLE_CUDA)
    , umpire::alloc::cuda_malloc_allocator
    , umpire::alloc::cuda_malloc_managed_allocator
    , umpire::alloc::cuda_pinned_allocator
#endif
#if defined(UMPIRE_ENABLE_HIP)
    , umpire::alloc::hip_malloc_allocator
    , umpire::alloc::hip_pinned_allocator
#endif
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
    , umpire::alloc::omp_target_allocator
#endif
>;

INSTANTIATE_TYPED_TEST_SUITE_P(_, memory_allocator_test, test_types,);
