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

namespace {

void run_test(const std::string& resource_name, const std::string& alloc_name,
              const umpire::strategy::GranularityController::Granularity gran)
{
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator resource;
  umpire::Allocator allocator;
  const size_t size{53};

  ASSERT_NO_THROW(resource = rm.getAllocator(resource_name));
  ASSERT_NO_THROW(allocator = rm.makeAllocator<umpire::strategy::GranularityController>(alloc_name, resource, gran));
  ASSERT_NO_THROW(allocator.deallocate(allocator.allocate(size)));
}

} // namespace

class GranularityController : public ::testing::TestWithParam<const char*> {
};

TEST_P(GranularityController, CourseGrain)
{
  run_test(std::string{GetParam()}, std::string{GetParam()} + "_CoarseGrainAllocator",
           umpire::strategy::GranularityController::Granularity::CoarseGrainedCoherence);
}

TEST_P(GranularityController, FineGrain)
{
  run_test(std::string{GetParam()}, std::string{GetParam()} + "_FineGrainAllocator",
           umpire::strategy::GranularityController::Granularity::FineGrainedCoherence);
}

const char* resource_names[3] = {"UM", "DEVICE", "PINNED"};

INSTANTIATE_TEST_SUITE_P(GranularityControllerTests, GranularityController, ::testing::ValuesIn(resource_names));
