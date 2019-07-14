//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "umpire/resource/DefaultMemoryResource.hpp"

struct TestAllocator
{
  void* allocate(std::size_t bytes)
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
  auto alloc = std::make_shared<umpire::resource::DefaultMemoryResource<TestAllocator> >(umpire::Platform::cpu, "TEST", 0, umpire::MemoryResourceTraits{});

  SUCCEED();
}

TEST(DefaultMemoryResource, AllocateDeallocate)
{
  auto alloc = std::make_shared<umpire::resource::DefaultMemoryResource<TestAllocator> >(umpire::Platform::cpu, "TEST", 0, umpire::MemoryResourceTraits{});
  double* pointer = (double*)alloc->allocate(10*sizeof(double));
  ASSERT_NE(pointer, nullptr);

  alloc->deallocate(pointer);
}
