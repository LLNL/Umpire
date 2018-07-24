
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
#include "umpire/Umpire.hpp"

class FreeFunctionsTest :
  public ::testing::TestWithParam<std::string> 
{
};

TEST_P(FreeFunctionsTest, DefaultMallocFree)
{
  double* test_alloc;
 
  ASSERT_NO_THROW(
     test_alloc = static_cast<double*>(umpire::malloc(100*sizeof(double)))
  );

  ASSERT_NE(nullptr, test_alloc);

  ASSERT_NO_THROW(
      umpire::free(test_alloc)
  );
}

TEST_P(FreeFunctionsTest, SetDefaultAndMallocFree)
{
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_NO_THROW(
      rm.setDefaultAllocator(
        rm.getAllocator(GetParam()))
  );

  double* test_alloc;
 
  ASSERT_NO_THROW(
     test_alloc = static_cast<double*>(umpire::malloc(100*sizeof(double)))
  );

  ASSERT_NE(nullptr, test_alloc);

  ASSERT_NO_THROW(
      umpire::free(test_alloc)
  );
}

const std::string allocators[] = {
  "HOST",
#if defined(UMPIRE_ENABLE_CUDA)
  , "DEVICE"
  , "UM"
  , "PINNED"
#endif
};

INSTANTIATE_TEST_CASE_P(
  FreeFunctions,
  FreeFunctionsTest,
  ::testing::ValuesIn(allocators));
