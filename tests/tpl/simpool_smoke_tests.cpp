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

#include "umpire/tpl/simpool/FixedSizePool.hpp"
#include "umpire/tpl/simpool/DynamicSizePool.hpp"
#include "umpire/tpl/simpool/StdAllocator.hpp"

TEST(Simpool, FixedPool)
{
  pool = FixedPool<int, StdAllocator>{};

  int* i = pool.allocate();

  pool.deallocate(i);

  ASSERT_TRUE();
}
