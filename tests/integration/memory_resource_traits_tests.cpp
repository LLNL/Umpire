//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
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

#include "umpire/config.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/resource/MemoryResourceTraits.hpp"

TEST(Traits, DDR) {
  umpire::ResourceManager& rm = umpire::ResourceManager::getInstance();

  using namespace umpire::resource;

  MemoryResourceTraits traits;
  traits.kind = MemoryResourceTraits::memory_type::DDR;

  auto allocator = rm.getAllocator(traits);

  float* data = static_cast<float*>(allocator.allocate(sizeof(float)*1024));

  for (int i = 0; i < 1024; i++) {
    data[i] = 3.14;
  }

  for (int i = 0; i < 1024; i++) {
    EXPECT_FLOAT_EQ(data[i], 3.14);
  }
}
