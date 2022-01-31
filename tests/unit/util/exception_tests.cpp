//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#include "umpire/util/error.hpp"
#include "umpire/util/Macros.hpp"

TEST(Exception, ThrowException)
{
  ASSERT_THROW(throw umpire::runtime_error("Test Exception", __FILE__, __LINE__), umpire::runtime_error);
  ASSERT_THROW(throw umpire::out_of_memory_error("Test OOM Exception", __FILE__, __LINE__), umpire::out_of_memory_error);
}

// TEST(Exception, ThrowFromErrorMacro)
// {
//   ASSERT_THROW({
//     UMPIRE_ERROR(runtime_error, "Test Exception")
//   }, umpire::runtime_error);
//   ASSERT_THROW({
//     UMPIRE_ERROR(out_of_memory_error, "Test OOM Exception"), 
//   }, umpire::out_of_memory_error);
// }
