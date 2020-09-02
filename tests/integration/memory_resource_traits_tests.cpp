//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

using umpire::MemoryResourceTraits;

TEST(MemoryResourceTraitsTest, HOST_Resource)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc_one = rm.getAllocator("HOST");
  auto alloc_two = rm.makeAllocator<umpire::strategy::SizeLimiter>(
      "HOST_Limiter", alloc_one, 1024);

  double* data =
      static_cast<double*>(alloc_one.allocate(1024 * sizeof(double)));

  ASSERT_EQ(MemoryResourceTraits::resource_type::HOST,
            alloc_two.getAllocationStrategy()->getTraits().resource);

  ASSERT_THROW(alloc_two.deallocate(data), umpire::util::Exception);

  ASSERT_NO_THROW(alloc_one.deallocate(data));
}

#if defined(UMPIRE_ENABLE_FILE_RESOURCE)
TEST(MemoryResourceTraitsTest, FILE_Resource)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc_one = rm.getAllocator("FILE");
  auto alloc_two = rm.makeAllocator<umpire::strategy::SizeLimiter>(
      "FILE_Limiter", alloc_one, 1024);

  double* data =
      static_cast<double*>(alloc_one.allocate(1024 * sizeof(double)));

  ASSERT_EQ(MemoryResourceTraits::resource_type::FILE,
            alloc_two.getAllocationStrategy()->getTraits().resource);

  ASSERT_THROW(alloc_two.deallocate(data), umpire::util::Exception);

  ASSERT_NO_THROW(alloc_one.deallocate(data));
}
#endif
#if defined(UMPIRE_ENABLE_DEVICE)
TEST(MemoryResourceTraitsTest, DEVICE_Resource)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc_one = rm.getAllocator("DEVICE");
  auto alloc_two = rm.makeAllocator<umpire::strategy::SizeLimiter>(
      "DEVICE_Limiter", alloc_one, 1024);

  double* data =
      static_cast<double*>(alloc_one.allocate(1024 * sizeof(double)));

  ASSERT_EQ(MemoryResourceTraits::resource_type::DEVICE,
            alloc_two.getAllocationStrategy()->getTraits().resource);

  ASSERT_THROW(alloc_two.deallocate(data), umpire::util::Exception);

  ASSERT_NO_THROW(alloc_one.deallocate(data));
}
#endif
#if defined(UMPIRE_ENABLE_CONST)
TEST(MemoryResourceTraitsTest, DEVICE_CONST_Resource)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc_one = rm.getAllocator("DEVICE_CONST");
  auto alloc_two = rm.makeAllocator<umpire::strategy::SizeLimiter>(
      "DEVICE_CONST_Limiter", alloc_one, 1024);

  double* data =
      static_cast<double*>(alloc_one.allocate(1024 * sizeof(double)));

  ASSERT_EQ(MemoryResourceTraits::resource_type::DEVICE_CONST,
            alloc_two.getAllocationStrategy()->getTraits().resource);

  ASSERT_THROW(alloc_two.deallocate(data), umpire::util::Exception);

  ASSERT_NO_THROW(alloc_one.deallocate(data));
}
#endif
#if defined(UMPIRE_ENABLE_PINNED)
TEST(MemoryResourceTraitsTest, PINNED_Resource)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc_one = rm.getAllocator("PINNED");
  auto alloc_two = rm.makeAllocator<umpire::strategy::SizeLimiter>(
      "PINNED_Limiter", alloc_one, 1024);

  double* data =
      static_cast<double*>(alloc_one.allocate(1024 * sizeof(double)));

  ASSERT_EQ(MemoryResourceTraits::resource_type::PINNED,
            alloc_two.getAllocationStrategy()->getTraits().resource);

  ASSERT_THROW(alloc_two.deallocate(data), umpire::util::Exception);

  ASSERT_NO_THROW(alloc_one.deallocate(data));
}
#endif
#if defined(UMPIRE_ENABLE_UM)
TEST(MemoryResourceTraitsTest, UM_Resource)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc_one = rm.getAllocator("UM");
  auto alloc_two = rm.makeAllocator<umpire::strategy::SizeLimiter>(
      "UM_Limiter", alloc_one, 1024);

  double* data =
      static_cast<double*>(alloc_one.allocate(1024 * sizeof(double)));

  ASSERT_EQ(MemoryResourceTraits::resource_type::UM,
            alloc_two.getAllocationStrategy()->getTraits().resource);

  ASSERT_THROW(alloc_two.deallocate(data), umpire::util::Exception);

  ASSERT_NO_THROW(alloc_one.deallocate(data));
}
#endif
