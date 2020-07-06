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

#include "gtest/gtest.h"

TEST(ResourceTest, AllocateDeallocate)
{
  auto alloc = std::make_shared<umpire::resource::FileMemoryResource>(umpire::Platform::host, "TEST", 0, umpire::MemoryResourceTraits{});
  auto pointer = alloc->allocate(sysconf(_SC_PAGE_SIZE) + 5000);
  ASSERT_NE(pointer, nullptr);

  alloc->deallocate(pointer);
}

TEST(ResourceTest, ZeroFile)
{
  auto alloc = std::make_shared<umpire::resource::FileMemoryResource>(umpire::Platform::host, "TEST", 0, umpire::MemoryResourceTraits{});
  ASSERT_THROW(alloc->allocate(0), umpire::util::Exception);
}

TEST(ResourceTest, VeryLargeFile)
{
  auto alloc = std::make_shared<umpire::resource::FileMemoryResource>(umpire::Platform::host, "TEST", 0, umpire::MemoryResourceTraits{});
  ASSERT_THROW(alloc->allocate((std::size_t)LDBL_MAX), umpire::util::Exception);
}

REGISTER_TYPED_TEST_SUITE_P(
    ResourceTest,
    Constructor, Allocate, getCurrentSize, getHighWatermark, getPlatform, getTraits);

INSTANTIATE_TYPED_TEST_SUITE_P(Mmap, ResourceTest, umpire::resource::FileMemoryResource,);