//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "resource_tests.hpp"
#include "umpire/resource/NullMemoryResource.hpp"
#include "umpire/util/Exception.hpp"

TYPED_TEST_P(ResourceTest, AccessNullResource)
{
  char* data = (char*)this->memory_resource->allocate(16);

  ASSERT_NE(nullptr, data);

  EXPECT_DEATH_IF_SUPPORTED(data[0] = 0, ".*");
}

REGISTER_TYPED_TEST_SUITE_P(ResourceTest, Constructor, Allocate, getCurrentSize,
                            getHighWatermark, getPlatform, getTraits,
                            AccessNullResource);

INSTANTIATE_TYPED_TEST_SUITE_P(Null, ResourceTest,
                               umpire::resource::NullMemoryResource, );
