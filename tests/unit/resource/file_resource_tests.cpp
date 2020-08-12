//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "gtest/gtest.h"
#include "resource_tests.hpp"
#include "umpire/resource/FileMemoryResource.hpp"
#include "umpire/util/Exception.hpp"

TYPED_TEST_P(ResourceTest, AllocateDeallocate)
{
  auto pointer_1 =
      this->memory_resource->allocate(sysconf(_SC_PAGE_SIZE) + 5000);
  ASSERT_NE(pointer_1, nullptr);

  auto pointer_2 =
      this->memory_resource->allocate(sysconf(_SC_PAGE_SIZE) - 1010);
  ASSERT_NE(pointer_2, nullptr);

  auto pointer_3 =
      this->memory_resource->allocate(sysconf(_SC_PAGE_SIZE) + 1010);
  ASSERT_NE(pointer_3, nullptr);

  this->memory_resource->deallocate(pointer_1);
  this->memory_resource->deallocate(pointer_2);
  this->memory_resource->deallocate(pointer_3);
}

TYPED_TEST_P(ResourceTest, ZeroFile)
{
  ASSERT_THROW(this->memory_resource->allocate(0), umpire::util::Exception);
}

TYPED_TEST_P(ResourceTest, LargeFile)
{
  std::size_t* ptr = nullptr;
  ASSERT_NO_THROW(ptr = (std::size_t*)this->memory_resource->allocate(
                      10000000000ULL * sizeof(std::size_t)));
  this->memory_resource->deallocate(ptr);
}

TYPED_TEST_P(ResourceTest, MmapFile)
{
  std::size_t* ptr = nullptr;
  ASSERT_NO_THROW(ptr = (std::size_t*)this->memory_resource->allocate(
                      1000000000ULL * sizeof(std::size_t)));

  std::size_t* start = ptr;
  for (int i = 0; i <= 9; i++) {
    *start = (size_t)i;
    start += sizeof(size_t);
  }
  start = ptr;
  for (int i = 0; i <= 9; i++) {
    if ((std::size_t)i != *start) {
      FAIL();
    }
    start += sizeof(size_t);
  }

  this->memory_resource->deallocate(ptr);
}

REGISTER_TYPED_TEST_SUITE_P(ResourceTest, Constructor, Allocate, getCurrentSize,
                            getHighWatermark, getPlatform, getTraits,
                            AllocateDeallocate, ZeroFile, LargeFile, MmapFile);

INSTANTIATE_TYPED_TEST_SUITE_P(Mmap, ResourceTest,
                               umpire::resource::FileMemoryResource, );
