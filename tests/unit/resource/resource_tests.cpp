//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/util/Platform.hpp"

struct TestAllocator {
  void* allocate(std::size_t bytes)
  {
    return ::malloc(bytes);
  }

  void deallocate(void* ptr)
  {
    ::free(ptr);
  }

  bool isAccessible(umpire::Platform p)
  {
    if (p == umpire::Platform::host)
      return true;
    else
      return false;
  }
};

TEST(DefaultMemoryResource, Constructor)
{
  auto alloc = std::make_shared<umpire::resource::DefaultMemoryResource<TestAllocator>>(
      umpire::Platform::host, "TEST", 0, umpire::MemoryResourceTraits{});

  SUCCEED();
}

TEST(DefaultMemoryResource, AllocateDeallocate)
{
  auto alloc = std::make_shared<umpire::resource::DefaultMemoryResource<TestAllocator>>(
      umpire::Platform::host, "TEST", 0, umpire::MemoryResourceTraits{});
  double* pointer = (double*)alloc->allocate(10 * sizeof(double));
  ASSERT_NE(pointer, nullptr);

  alloc->deallocate(pointer, 0);
}
