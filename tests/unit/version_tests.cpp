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

#include "umpire/config.hpp"
#include "umpire/Umpire.hpp"

TEST(Version, MajorVersion)
{
  ASSERT_EQ(
      UMPIRE_VERSION_MAJOR,
      umpire::get_major_version());
}

TEST(Version, MinorVersion)
{
  ASSERT_EQ(
      UMPIRE_VERSION_MINOR,
      umpire::get_minor_version());
}

TEST(Version, PatchVersion)
{
  ASSERT_EQ(
      UMPIRE_VERSION_PATCH,
      umpire::get_patch_version());
}
