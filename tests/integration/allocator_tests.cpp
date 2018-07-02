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

#include "umpire/resource/MemoryResourceTypes.hpp"

TEST(Allocator, HostAllocator)
{
  auto &rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("HOST");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));
  ASSERT_NE(nullptr, test_alloc);
  allocator.deallocate(test_alloc);

  test_alloc = static_cast<double*>(allocator.allocate(0*sizeof(double)));
  allocator.deallocate(test_alloc);
}

TEST(Allocator, HostAllocatorType)
{
  auto &rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator(umpire::resource::Host);
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);

  allocator.deallocate(test_alloc);
}

TEST(Allocator, HostAllocatorReference)
{
  auto &rm = umpire::ResourceManager::getInstance();
  umpire::Allocator *p;

  p = new umpire::Allocator(rm.getAllocator("HOST"));

  double* test_alloc = static_cast<double*>(p->allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);

  p->deallocate(test_alloc);

  delete p;
}

TEST(Allocator, HostAllocatorSize)
{
  auto &rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("HOST");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_EQ((100*sizeof(double)), allocator.getSize(test_alloc));

  allocator.deallocate(test_alloc);

  ASSERT_ANY_THROW(allocator.getSize(test_alloc));
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST(Allocator, DeviceAllocator)
{
  auto &rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("DEVICE");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);
}

TEST(Allocator, DeviceAllocatorReference)
{
  auto &rm = umpire::ResourceManager::getInstance();
  umpire::Allocator *p;

  p = new umpire::Allocator(rm.getAllocator("DEVICE"));

  double* test_alloc = static_cast<double*>(p->allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);

  p->deallocate(test_alloc);

  delete p;
}

TEST(Allocator, DeviceAllocatorSize)
{
  auto &rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("DEVICE");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_EQ((100*sizeof(double)), allocator.getSize(test_alloc));

  allocator.deallocate(test_alloc);

  ASSERT_ANY_THROW(allocator.getSize(test_alloc));
}

TEST(Allocator, UmAllocator)
{
  auto &rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("UM");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);
}

TEST(Allocator, UmAllocatorReference)
{
  auto &rm = umpire::ResourceManager::getInstance();
  umpire::Allocator *p;

  p = new umpire::Allocator(rm.getAllocator("UM"));

  double* test_alloc = static_cast<double*>(p->allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);

  p->deallocate(test_alloc);

  delete p;
}

TEST(Allocator, UmAllocatorSize)
{
  auto &rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("UM");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_EQ((100*sizeof(double)), allocator.getSize(test_alloc));

  allocator.deallocate(test_alloc);

  ASSERT_ANY_THROW(allocator.getSize(test_alloc));
}

TEST(Allocator, PinnedAllocator)
{
  auto &rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("PINNED");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);
}

TEST(Allocator, PinnedAllocatorReference)
{
  auto &rm = umpire::ResourceManager::getInstance();
  umpire::Allocator *p;

  p = new umpire::Allocator(rm.getAllocator("PINNED"));

  double* test_alloc = static_cast<double*>(p->allocate(100*sizeof(double)));

  ASSERT_NE(nullptr, test_alloc);

  p->deallocate(test_alloc);

  delete p;
}

TEST(Allocator, PinnedAllocatorSize)
{
  auto &rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("PINNED");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  ASSERT_EQ((100*sizeof(double)), allocator.getSize(test_alloc));

  allocator.deallocate(test_alloc);

  ASSERT_ANY_THROW(allocator.getSize(test_alloc));
}

#endif

TEST(Allocator, Deallocate)
{
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator allocator = rm.getAllocator("HOST");
  double* test_alloc = static_cast<double*>(allocator.allocate(100*sizeof(double)));

  rm.deallocate(test_alloc);

  SUCCEED();
}

TEST(Allocator, DeallocateThrow)
{
  auto& rm = umpire::ResourceManager::getInstance();

  double* ptr = new double[20];
  ASSERT_ANY_THROW(rm.deallocate(ptr));
}

TEST(Allocator, Name)
{
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator alloc = rm.getAllocator("HOST");

  ASSERT_EQ(alloc.getName(), "HOST");
}

TEST(Allocator, Id)
{
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator alloc = rm.getAllocator("HOST");
  int id = alloc.getId();
  ASSERT_GE(id, 0);

  auto allocator_by_id = rm.getAllocator(id);

  ASSERT_EQ(alloc.getAllocationStrategy(), allocator_by_id.getAllocationStrategy());
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST(Allocator, IdUnique)
{
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::Allocator host_alloc = rm.getAllocator("HOST");
  umpire::Allocator device_alloc = rm.getAllocator("DEVICE");

  int host_id = host_alloc.getId();
  int device_id = device_alloc.getId();

  ASSERT_NE(host_id, device_id);
}
#endif

TEST(Allocator, isRegistered)
{
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_TRUE(rm.isAllocatorRegistered("HOST"));
  ASSERT_FALSE(rm.isAllocatorRegistered("BANANAS"));
}

TEST(Allocator, registerAllocator)
{
  auto& rm = umpire::ResourceManager::getInstance();

  rm.registerAllocator("my_host_allocator_copy", rm.getAllocator("HOST"));

  ASSERT_EQ(rm.getAllocator("HOST").getAllocationStrategy(), 
      rm.getAllocator("my_host_allocator_copy").getAllocationStrategy());

  ASSERT_ANY_THROW(
      rm.registerAllocator("HOST", rm.getAllocator("my_host_allocator_copy")));
}
