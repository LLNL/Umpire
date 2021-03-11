//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

TEST(SharedMemory, DefaultTraits)
{
  auto traits{umpire::get_default_resource_traits("SHARED")};
  ASSERT_EQ(traits.scope, umpire::MemoryResourceTraits::shared_scope::node);
  ASSERT_EQ(traits.resource, umpire::MemoryResourceTraits::resource_type::shared);
}
