//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"

TEST(Version, MajorVersion)
{
  ASSERT_EQ(UMPIRE_VERSION_MAJOR, umpire::get_major_version());
}

TEST(Version, MinorVersion)
{
  ASSERT_EQ(UMPIRE_VERSION_MINOR, umpire::get_minor_version());
}

TEST(Version, PatchVersion)
{
  ASSERT_EQ(UMPIRE_VERSION_PATCH, umpire::get_patch_version());
}

TEST(Version, RcVersion)
{
  ASSERT_EQ(UMPIRE_VERSION_RC, umpire::get_rc_version());
}
