//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "gtest/gtest-death-test.h"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

using cPlatform = camp::resources::Platform;
using umpire::MemoryResourceTraits;

class AllocatorAccessibilityTest : public ::testing::TestWithParam<std::string> {
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
  double m_size = 1024;
};

TEST_P(AllocatorAccessibilityTest, AccessibilityFromHost)
{
  ::testing::FLAGS_gtest_death_test_style = "threadsafe"; 
  if(is_accessible(cPlatform::host, *m_allocator)) {
    double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
    ASSERT_NO_THROW(data[0] = m_size*m_size);
  } else {
    double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
    ASSERT_DEATH(data[0] = m_size*m_size, "");
  }
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST_P(AllocatorAccessibilityTest, AccessibilityFromCuda)
{
  ::testing::FLAGS_gtest_death_test_style = "threadsafe"; 
  if(is_accessible(cPlatform::cuda, *m_allocator)) {
    double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
    ASSERT_NO_THROW(data[0] = m_size*m_size);
  //} else if (m_allocator->getAllocationStrategy()->getTraits().resource == MemoryResourceTraits::resource_type::FILE) {
  //  double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
  //  ASSERT_NO_THROW(data[0] = m_size*m_size);
  } else {
   // double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
   // ASSERT_DEATH(data[0] = m_size*m_size, "");
   std::cout << "Hi " << std::endl <<std::endl;
  }
}
#endif
/*
#if defined(UMPIRE_ENABLE_HIP)
TEST_P(AllocatorAccessibilityTest, AccessibilityFromHip)
{
  ::testing::FLAGS_gtest_death_test_style = "threadsafe"; 
  if(is_accessible(cPlatform::hip, *m_allocator)) {
    double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
    ASSERT_NO_THROW(data[0] = m_size*m_size);
  } else {
    double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
    ASSERT_DEATH(data[0] = m_size*m_size, "");
  }
}
#endif

TEST_P(AllocatorAccessibilityTest, AccessibilityFromUndefined)
{
  ::testing::FLAGS_gtest_death_test_style = "threadsafe"; 
  if(is_accessible(cPlatform::undefined, *m_allocator)) {
    double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
    ASSERT_NO_THROW(data[0] = m_size*m_size);
  } else {
    double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
    ASSERT_DEATH(data[0] = m_size*m_size, "");
  }
}
*/
std::vector<std::string> get_allocators()
{
  auto& rm = umpire::ResourceManager::getInstance();
  std::vector<std::string> available_allocators = rm.getResourceNames();
  for(auto a : available_allocators)
    std::cout << a << " ";
  return available_allocators;
}

INSTANTIATE_TEST_SUITE_P(Allocators, AllocatorAccessibilityTest, ::testing::ValuesIn(get_allocators()));
