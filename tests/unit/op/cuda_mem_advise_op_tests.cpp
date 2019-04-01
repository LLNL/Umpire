//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#include "umpire/op/MemoryOperationRegistry.hpp"
#include "umpire/util/AllocationRecord.hpp"

TEST(CudaAdviseAccessedBy, Find)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  ASSERT_NO_THROW(
      op_registry.find(
      "ACCESSED_BY",
      strategy,
      strategy));
}

TEST(CudaAdviseAccessedBy, Apply)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  auto advice_operation = op_registry.find(
      "ACCESSED_BY",
      strategy,
      strategy);

  float* data = static_cast<float*>(allocator.allocate(1024*sizeof(float)));

  ASSERT_NO_THROW(
    advice_operation->apply(
      data,
      nullptr, // AllocationRecord* is unused
      0, // val is unused
      1024*sizeof(float)));
}

TEST(CudaAdvisePreferredLocation, Find)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  ASSERT_NO_THROW(
      op_registry.find(
      "PREFERRED_LOCATION",
      strategy,
      strategy));
}

TEST(CudaAdvisePreferredLocation, Apply)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  auto advice_operation = op_registry.find(
      "PREFERRED_LOCATION",
      strategy,
      strategy);

  float* data = static_cast<float*>(allocator.allocate(1024*sizeof(float)));
  auto record = new umpire::util::AllocationRecord{data, 1024*sizeof(float), strategy};

  ASSERT_NO_THROW(
    advice_operation->apply(
      data,
      record,
      0, // val is unused
      1024*sizeof(float)));

  allocator.deallocate(data);
}

TEST(CudaAdvisePreferredLocation, ApplyHost)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  auto advice_operation = op_registry.find(
      "PREFERRED_LOCATION",
      strategy,
      strategy);

  float* data = static_cast<float*>(allocator.allocate(1024*sizeof(float)));
  auto record = new umpire::util::AllocationRecord{data, 1024*sizeof(float), strategy};

  ASSERT_NO_THROW(
    advice_operation->apply(
      data,
      record,
      0, // val is unused
      1024*sizeof(float)));

  allocator.deallocate(data);
  delete record;
}

TEST(CudaAdviseReadMostly, Find)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  ASSERT_NO_THROW(
      op_registry.find(
      "READ_MOSTLY",
      strategy,
      strategy));
}

TEST(CudaAdviseReadMostly, Apply)
{
  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto advice_operation = op_registry.find(
      "READ_MOSTLY",
      strategy,
      strategy);

  float* data = static_cast<float*>(allocator.allocate(1024*sizeof(float)));

  ASSERT_NO_THROW(
    advice_operation->apply(
      data,
      nullptr, // AllocationRecord* is unused
      0, // val is unused
      1024*sizeof(float)));
}
