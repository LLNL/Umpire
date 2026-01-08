//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/ResourceManager.hpp"
#include "umpire/config.hpp"
#include "umpire/op/MemoryOperationRegistry.hpp"
#include "umpire/util/AllocationRecord.hpp"

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)

TEST(MemAdviseAccessedBy, Find)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  ASSERT_NO_THROW(op_registry.find("SET_ACCESSED_BY", strategy, strategy));
}

TEST(MemAdviseAccessedBy, Apply)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  auto advice_operation = op_registry.find("SET_ACCESSED_BY", strategy, strategy);

  float* data = static_cast<float*>(allocator.allocate(1024 * sizeof(float)));

  ASSERT_NO_THROW(advice_operation->apply(data,
                                          nullptr, // AllocationRecord* is unused
                                          0,       // val is unused
                                          1024 * sizeof(float)));

  allocator.deallocate(data);
}

TEST(MemAdvisePreferredLocation, Find)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  ASSERT_NO_THROW(op_registry.find("SET_PREFERRED_LOCATION", strategy, strategy));
}

TEST(MemAdvisePreferredLocation, Apply)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  auto advice_operation = op_registry.find("SET_PREFERRED_LOCATION", strategy, strategy);

  float* data = static_cast<float*>(allocator.allocate(1024 * sizeof(float)));
  auto record = new umpire::util::AllocationRecord{data, 1024 * sizeof(float), strategy};

  ASSERT_NO_THROW(advice_operation->apply(data, record,
                                          0, // val is unused
                                          1024 * sizeof(float)));

  allocator.deallocate(data);
  delete record;
}

TEST(MemAdvisePreferredLocation, ApplyHost)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  auto advice_operation = op_registry.find("SET_PREFERRED_LOCATION", strategy, strategy);

  float* data = static_cast<float*>(allocator.allocate(1024 * sizeof(float)));
  auto record = new umpire::util::AllocationRecord{data, 1024 * sizeof(float), strategy};

  ASSERT_NO_THROW(advice_operation->apply(data, record,
                                          0, // val is unused
                                          1024 * sizeof(float)));

  allocator.deallocate(data);
  delete record;
}

TEST(MemAdviseReadMostly, Find)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  ASSERT_NO_THROW(op_registry.find("SET_READ_MOSTLY", strategy, strategy));
}

TEST(MemAdviseReadMostly, Apply)
{
  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto advice_operation = op_registry.find("SET_READ_MOSTLY", strategy, strategy);

  float* data = static_cast<float*>(allocator.allocate(1024 * sizeof(float)));

  ASSERT_NO_THROW(advice_operation->apply(data,
                                          nullptr, // AllocationRecord* is unused
                                          0,       // val is unused
                                          1024 * sizeof(float)));

  allocator.deallocate(data);
}

#if defined(UMPIRE_ENABLE_HIP)
// Test HIP-specific operations that don't exist in CUDA
TEST(MemAdviseCoarseGrain, Find)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  ASSERT_NO_THROW(op_registry.find("SET_COARSE_GRAIN", strategy, strategy));
}

TEST(MemAdviseCoarseGrain, Apply)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("UM");
  auto strategy = allocator.getAllocationStrategy();

  auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  auto advice_operation = op_registry.find("SET_COARSE_GRAIN", strategy, strategy);

  float* data = static_cast<float*>(allocator.allocate(1024 * sizeof(float)));

  ASSERT_NO_THROW(advice_operation->apply(data,
                                          nullptr, // AllocationRecord* is unused
                                          0,       // val is unused
                                          1024 * sizeof(float)));

  allocator.deallocate(data);
}
#endif // defined(UMPIRE_ENABLE_HIP)

#endif // defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)