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
#include "gtest/gtest.h"

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/DynamicPool.hpp"

TEST(SimpoolStrategy, Host)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "host_simpool", rm.getAllocator("HOST"));

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "host_simpool");
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST(SimpoolStrategy, Device)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "device_simpool", rm.getAllocator("DEVICE"));

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "device_simpool");
}

TEST(SimpoolStrategy, UM)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "um_simpool", rm.getAllocator("UM"));

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "um_simpool");
}
#endif

TEST(MonotonicStrategy, Host)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
      "host_monotonic_pool", 65536, rm.getAllocator("HOST"));

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "host_monotonic_pool");
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST(MonotonicStrategy, Device)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
      "device_monotonic_pool", 65536, rm.getAllocator("DEVICE"));

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "device_monotonic_pool");
}

TEST(MonotonicStrategy, UM)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
      "um_monotonic_pool", 65536, rm.getAllocator("UM"));

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "um_monotonic_pool");
}
#endif

TEST(Allocator, Duplicate)
{
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_TRUE(rm.isAllocator("HOST"));

  ASSERT_ANY_THROW(
      rm.makeAllocator<umpire::strategy::DynamicPool>(
        "host_simpool", rm.getAllocator("HOST")));
}
