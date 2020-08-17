//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"

class FreeFunctionsTest : public ::testing::TestWithParam<std::string> {
};

TEST_P(FreeFunctionsTest, DefaultMallocFree)
{
  double* test_alloc;

  ASSERT_NO_THROW(
      test_alloc = static_cast<double*>(umpire::malloc(100 * sizeof(double))));

  ASSERT_NE(nullptr, test_alloc);

  ASSERT_NO_THROW(umpire::free(test_alloc));
}

TEST_P(FreeFunctionsTest, SetDefaultAndMallocFree)
{
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_NO_THROW(rm.setDefaultAllocator(rm.getAllocator(GetParam())));

  double* test_alloc;

  ASSERT_NO_THROW(
      test_alloc = static_cast<double*>(umpire::malloc(100 * sizeof(double))));

  ASSERT_NE(nullptr, test_alloc);

  ASSERT_NO_THROW(umpire::free(test_alloc));
}

const std::string allocators[] = {"HOST"
#if defined(UMPIRE_ENABLE_CUDA)
                                  ,
                                  "DEVICE",
                                  "UM",
                                  "PINNED"
#endif
#if defined(UMPIRE_ENABLE_HIP)
                                  ,
                                  "DEVICE",
                                  "PINNED"
#endif
};

INSTANTIATE_TEST_SUITE_P(FreeFunctions, FreeFunctionsTest,
                         ::testing::ValuesIn(allocators));
