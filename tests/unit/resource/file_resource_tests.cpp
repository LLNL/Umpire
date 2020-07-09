//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "resource_tests.hpp"

#include "umpire/resource/FileMemoryResource.hpp"

#include "umpire/util/Exception.hpp"

#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "gtest/gtest.h"

TEST(ResourceTest, AllocateDeallocate)
{
  auto alloc = std::make_shared<umpire::resource::FileMemoryResource>(umpire::Platform::host, "TEST", 0, umpire::MemoryResourceTraits{});
  auto pointer_1 = alloc->allocate(sysconf(_SC_PAGE_SIZE) + 5000);
  ASSERT_NE(pointer_1, nullptr);

  auto pointer_2 = alloc->allocate(sysconf(_SC_PAGE_SIZE) - 1010);
  ASSERT_NE(pointer_2, nullptr);

  auto pointer_3 = alloc->allocate(sysconf(_SC_PAGE_SIZE) + 1010);
  ASSERT_NE(pointer_3, nullptr);

  alloc->deallocate(pointer_1);
  alloc->deallocate(pointer_2);
  alloc->deallocate(pointer_3);
}

TEST(ResourceTest, ZeroFile)
{
  auto alloc = std::make_shared<umpire::resource::FileMemoryResource>(umpire::Platform::host, "TEST", 0, umpire::MemoryResourceTraits{});
  ASSERT_THROW(alloc->allocate(0), umpire::util::Exception);
}

TEST(ResourceTest, TooLargeForSystem)
{
  auto alloc = std::make_shared<umpire::resource::FileMemoryResource>(umpire::Platform::host, "TEST", 0, umpire::MemoryResourceTraits{});
  ASSERT_THROW(alloc->allocate( ULLONG_MAX ), umpire::util::Exception);
}

TEST(ResourceTest, LargeFile)
{
  auto alloc = std::make_shared<umpire::resource::FileMemoryResource>(umpire::Platform::host, "TEST", 0, umpire::MemoryResourceTraits{});
  std::size_t* ptr;
  ASSERT_NO_THROW(ptr = (std::size_t*) alloc->allocate( 10000000000ULL*sizeof(std::size_t) ) );
  alloc->deallocate(ptr);
}

TEST(ResourceTest, MmapFile)
{
  auto alloc = std::make_shared<umpire::resource::FileMemoryResource>(umpire::Platform::host, "TEST", 0, umpire::MemoryResourceTraits{});
  std::size_t* ptr;
  ASSERT_NO_THROW(ptr = (std::size_t*) alloc->allocate( 1000000000ULL*sizeof(std::size_t)));

  std::size_t* start = ptr;
  for(int i = 0; i <= 9; i++)
  {
    *start = (size_t)i; 
    start += sizeof(size_t);
  }
  start = ptr;
  for(int i = 0; i <= 9; i++)
  {
    if((std::size_t)i != *start){ FAIL(); }
    start += sizeof(size_t);
  }

  alloc->deallocate(ptr);
}

REGISTER_TYPED_TEST_SUITE_P(
    ResourceTest,
    Constructor, Allocate, getCurrentSize, getHighWatermark, getPlatform, getTraits);

INSTANTIATE_TYPED_TEST_SUITE_P(Mmap, ResourceTest, umpire::resource::FileMemoryResource,);