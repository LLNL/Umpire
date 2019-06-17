//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
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
}

TEST(DefaultMemoryResource, GetSize)
{
  auto alloc = std::make_shared<umpire::resource::DefaultMemoryResource<TestAllocator> >(umpire::Platform::cpu, "TEST", 0, umpire::MemoryResourceTraits{});
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
  auto alloc = std::make_shared<umpire::resource::DefaultMemoryResource<TestAllocator> >(umpire::Platform::cpu, "TEST", 0, umpire::MemoryResourceTraits{});
  ASSERT_EQ(alloc->getHighWatermark(), 0);

  double* pointer = (double*)alloc->allocate(10);
  double* pointer_two = (double*)alloc->allocate(30);

  alloc->deallocate(pointer);

  ASSERT_EQ(alloc->getHighWatermark(), 40);

  alloc->deallocate(pointer_two);
}
