//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "camp/camp.hpp"
#include "gtest/gtest.h"
#include "umpire/ResourceManager.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/ResourceAwarePool.hpp"

using namespace camp::resources;

TEST(ResourceAwarePoolTest, Construction)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool", rm.getAllocator("HOST"));

  EXPECT_EQ(pool.getCurrentSize(), 0);
}

TEST(ResourceAwarePoolTest, Check_States)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool2", rm.getAllocator("DEVICE"));

#if defined(UMPIRE_ENABLE_CUDA)
  Cuda d1;
#elif defined(UMPIRE_ENABLE_HIP)
  Hip d1;
#else
  Host d1;
#endif

  Resource r1{d1};

  void* ptr{pool.allocate(r1, 1024)};

  // EXPECT_EQ(pool.getResource(ptr), Cuda);

  pool.deallocate(ptr);
}
