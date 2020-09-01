//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_resource_tests_HPP
#define UMPIRE_resource_tests_HPP

#include <memory>

#include "gtest/gtest.h"
#include "umpire/util/Macros.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"
#include "umpire/util/Platform.hpp"

template <typename Resource>
class ResourceTest : public ::testing::Test {
 public:
  void SetUp() override
  {
    auto platform = umpire::Platform::undefined;
    const std::string name{"test resource"};
    const int id{0};
    auto traits = umpire::MemoryResourceTraits{};

    memory_resource = new Resource{platform, name, id, traits};
  }

  void TearDown() override
  {
    delete memory_resource;
  }

 protected:
  Resource* memory_resource;
};

TYPED_TEST_SUITE_P(ResourceTest);

TYPED_TEST_P(ResourceTest, Constructor)
{
  SUCCEED();
}

TYPED_TEST_P(ResourceTest, Allocate)
{
  void* data = this->memory_resource->allocate(64);

  ASSERT_NE(data, nullptr);

  this->memory_resource->deallocate(data);
}

TYPED_TEST_P(ResourceTest, getCurrentSize)
{
  auto size = this->memory_resource->getCurrentSize();
  ASSERT_GE(size, 0);
}

TYPED_TEST_P(ResourceTest, getHighWatermark)
{
  auto size = this->memory_resource->getHighWatermark();
  ASSERT_GE(size, 0);
}

TYPED_TEST_P(ResourceTest, getPlatform)
{
  auto platform = this->memory_resource->getPlatform();
  ASSERT_EQ(platform, umpire::Platform::undefined);
}

TYPED_TEST_P(ResourceTest, getTraits)
{
  auto traits = this->memory_resource->getTraits();

  UMPIRE_USE_VAR(traits);

  SUCCEED();
}

#endif // UMPIRE_resource_tests_HPP
