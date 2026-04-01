//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/null_resource.hpp"

#include <gtest/gtest.h>
#include <type_traits>

using namespace umpire::resource;

TEST(null_resource, default_behavior_throws_out_of_memory)
{
  null_resource<> mem("NULL_THROW");

  EXPECT_THROW(mem.allocate(64), umpire::out_of_memory_error);
}

TEST(null_resource, explicit_throw_behavior_throws_out_of_memory)
{
  null_resource<null_behavior::throw_exception> mem("NULL_THROW_EXPLICIT");

  EXPECT_THROW(mem.allocate(1), umpire::out_of_memory_error);
}

TEST(null_resource, return_nullptr_behavior_returns_nullptr)
{
  null_resource<null_behavior::return_nullptr> mem("NULL_PTR");

  EXPECT_EQ(mem.allocate(64), nullptr);
  EXPECT_EQ(mem.allocate(0), nullptr);
}

TEST(null_resource, deallocate_nullptr_is_safe_noop)
{
  null_resource<> mem("NULL_DEALLOC");

  EXPECT_NO_THROW(mem.deallocate(nullptr));
}

TEST(null_resource, deallocate_non_nullptr_is_safe_noop)
{
  null_resource<> mem("NULL_DEALLOC_NON_NULL");

  int value = 42;
  EXPECT_NO_THROW(mem.deallocate(&value));
}

TEST(null_resource, custom_instance_with_custom_name)
{
  null_resource<> mem("CUSTOM_NULL");

  EXPECT_EQ(mem.get_name(), "CUSTOM_NULL");
}

TEST(null_resource, platform_type_is_undefined)
{
  null_resource<> mem("PLATFORM_NULL");

  EXPECT_EQ(mem.get_platform(), umpire::resource::Platform::undefined);
}

TEST(null_resource, tracking_is_disabled)
{
  null_resource<> mem("TRACKING_NULL");

  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 0);

  EXPECT_THROW(mem.allocate(32), umpire::out_of_memory_error);

  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 0);
}

TEST(null_resource, return_nullptr_behavior_keeps_statistics_at_zero)
{
  null_resource<null_behavior::return_nullptr> mem("NULL_STATS");

  EXPECT_EQ(mem.allocate(128), nullptr);
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 0);
}

TEST(null_resource, convenience_aliases)
{
  static_assert(default_null_resource::tracking_enabled == false,
                "default_null_resource should have tracking disabled");
  static_assert(silent_null_resource::tracking_enabled == false,
                "silent_null_resource should have tracking disabled");
}

TEST(null_resource, type_traits)
{
  using throwing_null = null_resource<null_behavior::throw_exception>;
  using returning_null = null_resource<null_behavior::return_nullptr>;

  static_assert(std::is_same_v<throwing_null::platform, umpire::undefined_platform>,
                "Platform should be undefined_platform");
  static_assert(std::is_same_v<throwing_null::allocator_type, null_allocator>,
                "Allocator type should be null_allocator");
  static_assert(throwing_null::tracking_enabled == false,
                "tracking_enabled should be false");
  static_assert(returning_null::tracking_enabled == false,
                "tracking_enabled should be false");
}

TEST(null_resource, useful_for_error_path_testing)
{
  null_resource<> mem("ERROR_PATH_NULL");

  auto allocate_with_fallback = [&]() -> bool {
    try {
      (void)mem.allocate(256);
      return false;
    } catch (const umpire::out_of_memory_error&) {
      return true;
    }
  };

  EXPECT_TRUE(allocate_with_fallback());
}
