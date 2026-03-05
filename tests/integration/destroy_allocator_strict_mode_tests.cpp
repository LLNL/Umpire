//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/QuickPool.hpp"

// These tests require UMPIRE_STRICT_DESTRUCTION=1 to be set in the environment
// before the process starts, since the mode is determined by a static variable.
// The CMakeLists.txt sets this environment variable when running this test suite.

TEST(DestroyAllocatorStrictModeTest, ActiveAllocations)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("test_pool_strict", rm.getAllocator("HOST"));
  void* ptr = pool.allocate(100);
  ASSERT_NE(nullptr, ptr);

  // Try to destroy with active allocations - should throw error in strict mode
  ASSERT_THROW(rm.destroyAllocator("test_pool_strict", false), umpire::runtime_error);

  // Clean up
  pool.deallocate(ptr);
  rm.destroyAllocator("test_pool_strict");
}

TEST(DestroyAllocatorStrictModeTest, ParentChildWarning)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto parent = rm.makeAllocator<umpire::strategy::QuickPool>("test_parent_strict", rm.getAllocator("HOST"));
  auto child = rm.makeAllocator<umpire::strategy::QuickPool>("test_child_strict", parent);

  // Try to destroy parent - should throw error in strict mode
  ASSERT_THROW(rm.destroyAllocator("test_parent_strict"), umpire::runtime_error);

  // Clean up in correct order
  rm.destroyAllocator("test_child_strict");
  rm.destroyAllocator("test_parent_strict");
}
