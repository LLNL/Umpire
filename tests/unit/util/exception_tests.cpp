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
#include "umpire/util/Exception.hpp"
#include "umpire/util/Macros.hpp"

#include "gtest/gtest.h"

TEST(Exception, ThrowException)
{
  ASSERT_THROW(throw umpire::util::Exception("Test Exception", __FILE__, __LINE__),
               umpire::util::Exception);
}

TEST(Exception, ThrowFromErrorMacro)
{
  ASSERT_THROW(UMPIRE_ERROR( "Test Exception" ),
      umpire::util::Exception);
}
