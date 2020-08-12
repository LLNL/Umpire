//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/ResourceManager.hpp"
#include "umpire/config.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

TEST(Traits, DDR)
{
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  using namespace umpire::resource;

  MemoryResourceTraits traits;
  traits.kind = MemoryResourceTraits::memory_type::DDR;

  auto allocator = rm.getAllocator(traits);

  float* data = static_cast<float*>(allocator.allocate(sizeof(float) * 1024));

  for (int i = 0; i < 1024; i++) {
    data[i] = 3.14;
  }

  for (int i = 0; i < 1024; i++) {
    EXPECT_FLOAT_EQ(data[i], 3.14);
  }
}
