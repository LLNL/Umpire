//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"
#include "umpire/interface/umpire.h"

TEST(Version, MajorVersion)
{
  ASSERT_EQ(UMPIRE_VERSION_MAJOR, umpire_get_major_version());
}

TEST(Version, MinorVersion)
{
  ASSERT_EQ(UMPIRE_VERSION_MINOR, umpire_get_minor_version());
}

TEST(Version, PatchVersion)
{
  ASSERT_EQ(UMPIRE_VERSION_PATCH, umpire_get_patch_version());
}
