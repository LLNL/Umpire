//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/GranularityController.hpp"

TEST(AllocatorGranularity, UM_Coarse)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto resource = rm.getAllocator("UM");
  auto allocator = rm.makeAllocator<umpire::strategy::GranularityController>(
      "UM_CoarseGrainAllocator", resource,
      umpire::strategy::GranularityController::Granularity::CoarseGrainedCoherence);
  const size_t size{53};

  allocator.deallocate(allocator.allocate(size));
}

TEST(AllocatorGranularity, UM_Fine)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto resource = rm.getAllocator("UM");
  auto allocator = rm.makeAllocator<umpire::strategy::GranularityController>(
      "UM_FineGrainAllocator", resource, umpire::strategy::GranularityController::Granularity::CoarseGrainedCoherence);
  const size_t size{53};

  allocator.deallocate(allocator.allocate(size));
}

TEST(AllocatorGranularity, DEVICE_Coarse)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto resource = rm.getAllocator("DEVICE");
  auto allocator = rm.makeAllocator<umpire::strategy::GranularityController>(
      "DEVICE_CoarseGrainAllocator", resource,
      umpire::strategy::GranularityController::Granularity::CoarseGrainedCoherence);
  const size_t size{53};

  allocator.deallocate(allocator.allocate(size));
}

TEST(AllocatorGranularity, DEVICE_Fine)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto resource = rm.getAllocator("DEVICE");
  auto allocator = rm.makeAllocator<umpire::strategy::GranularityController>(
      "DEVICE_FineGrainAllocator", resource,
      umpire::strategy::GranularityController::Granularity::CoarseGrainedCoherence);
  const size_t size{53};

  allocator.deallocate(allocator.allocate(size));
}

TEST(AllocatorGranularity, PINNED_Coarse)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto resource = rm.getAllocator("PINNED");
  auto allocator = rm.makeAllocator<umpire::strategy::GranularityController>(
      "PINNED_CoarseGrainAllocator", resource,
      umpire::strategy::GranularityController::Granularity::CoarseGrainedCoherence);
  const size_t size{53};

  allocator.deallocate(allocator.allocate(size));
}

TEST(AllocatorGranularity, PINNED_Fine)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto resource = rm.getAllocator("PINNED");
  auto allocator = rm.makeAllocator<umpire::strategy::GranularityController>(
      "PINNED_FineGrainAllocator", resource,
      umpire::strategy::GranularityController::Granularity::CoarseGrainedCoherence);
  const size_t size{53};

  allocator.deallocate(allocator.allocate(size));
}
