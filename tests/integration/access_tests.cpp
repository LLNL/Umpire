//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"

#include "accessors.hpp"

class AccessTest : public ::testing::TestWithParam<std::string> {
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

  const std::size_t m_size = 1024;
};

TEST_P(AccessTest, VectorAdd)
{
  std::cout << "Running with allocator " << m_allocator->getName() << std::endl;

  double* a =
      static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
  double* b =
      static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
  double* c =
      static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
  ASSERT_NE(nullptr, a);
  ASSERT_NE(nullptr, b);
  ASSERT_NE(nullptr, c);

  umpire::forall(m_allocator->getAllocationStrategy()->getTraits().resource, 
    0,
    m_size,
    [=] __host__ __device__ (int i)  {
      b[i] = i;
      c[i] = 1.0;

      a[i] = b[i] + c[i];
  });
};

std::vector<std::string> allocator_strings() {
  std::vector<std::string> allocators;
  allocators.push_back("HOST");
#if defined(UMPIRE_ENABLE_DEVICE)
  allocators.push_back("DEVICE");
  auto& rm = umpire::ResourceManager::getInstance();
  for (int id = 0; id < rm.getNumDevices(); id++) {
    allocators.push_back(std::string{"DEVICE::" + std::to_string(id)});
  }
#endif
#if defined(UMPIRE_ENABLE_UM)
  allocators.push_back("UM");
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  allocators.push_back("PINNED");
#endif

  return allocators;
}

INSTANTIATE_TEST_SUITE_P(Access, AccessTest, ::testing::ValuesIn(allocator_strings()));