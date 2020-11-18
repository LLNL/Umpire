//////////////////////////////////////////////////////////////////////////////
//// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
//// project contributors. See the COPYRIGHT file for details.
////
//// SPDX-License-Identifier: (MIT)
////////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/ResourceManager.hpp"

// Needs to be in separate file so that resources are not initialized prior to
// reallocate call
TEST(Reallocate, Nullptr)
{
  auto& rm = umpire::ResourceManager::getInstance();
  constexpr std::size_t size = 1024;

  void* ptr{nullptr};
  EXPECT_NO_THROW({ ptr = rm.reallocate(ptr, size); });

  ASSERT_NE(nullptr, ptr);

  rm.deallocate(ptr);
}
