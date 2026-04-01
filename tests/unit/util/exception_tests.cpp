//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/error.hpp"
#include "umpire/util/error.hpp"

#include <type_traits>

TEST(Exception, ThrowException)
{
  static_assert(std::is_base_of_v<std::runtime_error, umpire::runtime_error>);
  static_assert(std::is_base_of_v<std::logic_error, umpire::logic_error>);
  static_assert(std::is_base_of_v<std::bad_alloc, umpire::out_of_memory>);
  static_assert(std::is_base_of_v<std::runtime_error, umpire::unknown_allocation>);

  ASSERT_THROW(throw umpire::runtime_error("Test Exception", __FILE__, __LINE__), umpire::runtime_error);
  ASSERT_THROW(throw umpire::logic_error("Test Logic Exception", __FILE__, __LINE__), umpire::logic_error);
  ASSERT_THROW(throw umpire::out_of_memory("Test OOM Exception", __FILE__, __LINE__), umpire::out_of_memory);
  ASSERT_THROW(throw umpire::unknown_allocation("Test Unknown Allocation", __FILE__, __LINE__),
               umpire::unknown_allocation);
}

TEST(Exception, LegacyCompatibilityAliases)
{
  ASSERT_THROW(throw umpire::out_of_memory_error("Legacy OOM", __FILE__, __LINE__), umpire::out_of_memory);
  ASSERT_THROW(throw umpire::unknown_pointer_error("Legacy Unknown Pointer", __FILE__, __LINE__),
               umpire::unknown_allocation);
}
