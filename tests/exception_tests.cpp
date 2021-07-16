//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/exception.hpp"
#include "umpire/detail/log.hpp"

#include "gtest/gtest.h"

TEST(exception, ThrowException)
{
  ASSERT_THROW(throw umpire::exception("Test Exception", __FILE__, __LINE__),
               umpire::exception);
}

TEST(exception, ThrowFromErrorMacro)
{
  ASSERT_THROW(UMPIRE_ERROR( "Test Exception" ),
      umpire::exception);
}
