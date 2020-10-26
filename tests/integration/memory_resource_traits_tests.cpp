//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

using myResource = umpire::MemoryResourceTraits::resource_type;

//
//This test confirms that when an allocator is created with a specific
//memory resource, that same memory resource can be quiered and returned
//
class MemoryResourceTraitsTest : public ::testing::TestWithParam<std::string> {
 public:
  virtual void SetUp()
  {
    auto& rm = umpire::ResourceManager::getInstance();
    
    m_allocator = new umpire::Allocator(rm.getAllocator(GetParam()));
    m_allocator_pool = new umpire::Allocator(rm.makeAllocator<umpire::strategy::QuickPool>
        ("pool_" + std::to_string(m_allocator->getId()), *m_allocator));
    
    m_resource = GetParam();
  }

  virtual void TearDown()
  {
    if(m_allocator)
      delete m_allocator;
    if(m_allocator_pool)
      delete m_allocator_pool;
  }

  umpire::Allocator* m_allocator;
  umpire::Allocator* m_allocator_pool;
  std::string m_resource;
};

umpire::MemoryResourceTraits::resource_type getResourceType(std::string resource)
{
  if(resource == "HOST")
    return myResource::host;
  else if(resource == "DEVICE")
    return myResource::device;
  else if(resource == "DEVICE_CONST")
    return myResource::device_const;
  else if(resource == "UM")
    return myResource::um;
  else if(resource == "PINNED")
    return myResource::pinned;
  else if(resource == "FILE")
    return myResource::file;
  else //unknown
    return myResource::unknown;
}

TEST_P(MemoryResourceTraitsTest, ResourceTraitTest)
{
  double* data =
      static_cast<double*>(m_allocator->allocate(1024 * sizeof(double)));

  myResource resource = getResourceType(m_resource);

  ASSERT_EQ(resource, m_allocator_pool->getAllocationStrategy()->getTraits().resource);

  ASSERT_EQ(m_allocator->getName(), m_resource);

  ASSERT_THROW(m_allocator_pool->deallocate(data), umpire::util::Exception);

  ASSERT_NO_THROW(m_allocator->deallocate(data));
}

//returns a vector of strings with the names of the
//memory resources currently available.
std::vector<std::string> memory_resource_strings()
{
  std::vector<std::string> resources;
  resources.push_back("HOST");
#if defined(UMPIRE_ENABLE_DEVICE)
  resources.push_back("DEVICE");
#endif
#if defined(UMPIRE_ENABLE_UM)
  resources.push_back("UM");
#endif
#if defined(UMPIRE_ENABLE_CONST)
  resources.push_back("DEVICE_CONST");
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  resources.push_back("PINNED");
#endif

  return resources;
}

INSTANTIATE_TEST_SUITE_P(MemoryResourceTraits, MemoryResourceTraitsTest,
                         ::testing::ValuesIn(memory_resource_strings()));
