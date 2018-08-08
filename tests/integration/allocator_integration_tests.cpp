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

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/resource/MemoryResourceTypes.hpp"

class AllocatorTest :
  public ::testing::TestWithParam< std::string >
{
  public:
  virtual void SetUp()
  {
      auto& rm = umpire::ResourceManager::getInstance();
      m_allocator = new umpire::Allocator(rm.getAllocator(GetParam()));
  }

  virtual void TearDown()
  {
    delete m_allocator;
  }

  umpire::Allocator* m_allocator;

  const size_t m_big = 16000;
  const size_t m_small = 8;
  const size_t m_nothing = 0;
};

TEST_P(AllocatorTest, AllocateDeallocateBig)
{
  double* data = static_cast<double*>(
    m_allocator->allocate(m_big*sizeof(double)));

  ASSERT_NE(nullptr, data);

  m_allocator->deallocate(data);
}

TEST_P(AllocatorTest, AllocateDeallocateSmall)
{
  double* data = static_cast<double*>(
    m_allocator->allocate(m_small*sizeof(double)));

  ASSERT_NE(nullptr, data);

  m_allocator->deallocate(data);
}

TEST_P(AllocatorTest, AllocateDeallocateNothing)
{
  double* data = static_cast<double*>(
    m_allocator->allocate(m_nothing*sizeof(double)));

  ASSERT_NE(nullptr, data);

  m_allocator->deallocate(data);
}

TEST_P(AllocatorTest, DeallocateNullptr)
{
  double* data = nullptr;

  ASSERT_NO_THROW(
  m_allocator->deallocate(data);
  );

  SUCCEED();
}

TEST_P(AllocatorTest, GetSize)
{
  const size_t size = m_big*sizeof(double);

  double* data = static_cast<double*>(
    m_allocator->allocate(size));

  ASSERT_EQ(size, m_allocator->getSize(data));

  m_allocator->deallocate(data);

  ASSERT_ANY_THROW(m_allocator->getSize(data));
}

TEST_P(AllocatorTest, GetName)
{
  ASSERT_EQ(m_allocator->getName(), GetParam());
}

TEST_P(AllocatorTest, GetById)
{
  auto& rm = umpire::ResourceManager::getInstance();

  int id = m_allocator->getId();
  ASSERT_GE(id, 0);

  auto allocator_by_id = rm.getAllocator(id);

  ASSERT_EQ(m_allocator->getAllocationStrategy(), allocator_by_id.getAllocationStrategy());

  ASSERT_THROW(
      rm.getAllocator(-25),
      umpire::util::Exception);
}

const std::string allocator_strings[] = {
  "HOST"
#if defined(UMPIRE_ENABLE_CUDA)
  , "DEVICE"
  , "UM"
  , "PINNED"
#endif
};

INSTANTIATE_TEST_CASE_P(
    Allocators,
    AllocatorTest,
    ::testing::ValuesIn(allocator_strings)
);

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

TEST(Allocator, GetSetDefault)
{
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_NO_THROW(
    auto alloc = rm.getDefaultAllocator();
  );

  ASSERT_NO_THROW(
    rm.setDefaultAllocator(rm.getDefaultAllocator());
  );
}

class AllocatorByResourceTest :
  public ::testing::TestWithParam< umpire::resource::MemoryResourceType >
{
  public:
  virtual void SetUp()
  {
      auto& rm = umpire::ResourceManager::getInstance();
      m_allocator = new umpire::Allocator(rm.getAllocator(GetParam()));
  }

  virtual void TearDown()
  {
    delete m_allocator;
  }

  umpire::Allocator* m_allocator;

  const size_t m_big = 16000;
  const size_t m_small = 8;
  const size_t m_nothing = 0;
};

TEST_P(AllocatorByResourceTest, AllocateDeallocate)
{
  double* data = static_cast<double*>(
    m_allocator->allocate(m_big*sizeof(double)));

  ASSERT_NE(nullptr, data);

  m_allocator->deallocate(data);
}

const umpire::resource::MemoryResourceType resource_types[] = {
  umpire::resource::Host
#if defined(UMPIRE_ENABLE_CUDA)
  , umpire::resource::Device
  , umpire::resource::UnifiedMemory
  , umpire::resource::PinnedMemory
#endif
};

INSTANTIATE_TEST_CASE_P(
    Resources,
    AllocatorByResourceTest,
    ::testing::ValuesIn(resource_types));
