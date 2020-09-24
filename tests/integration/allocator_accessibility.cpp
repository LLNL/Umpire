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

//#include "omp.h"

using cPlatform = camp::resources::Platform;
using myResource = umpire::MemoryResourceTraits::resource_type;

#if defined(UMPIRE_ENABLE_CUDA)
__global__ void tester(double* ptr, double m_size)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
   
  if (idx == 0) {
    ptr[0] = m_size * m_size;
  }
}

void test(double* ptr, double m_size)
{
  tester<<<1, 16>>>(ptr, m_size);
  cudaDeviceSynchronize();
}
#endif

#if defined(UMPIRE_ENABLE_HIP)
__global__ void tester(double* ptr, double m_size)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
   
  if (idx == 0) {
    ptr[0] = m_size * m_size;
  }
}

void test(double* ptr, double m_size)
{
  hipLaunchKernelGGL(tester, dim3(1), dim3(16), 0,0, ptr, m_size);
  hipDeviceSynchronize();
}
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
void test(double* ptr, double m_size)
{
  double* dev_ptr{static_cast<double*>(ptr)};

#pragma omp target is_device_ptr(data_ptr) device(device)
#pragma omp teams distribute parallel for schedule(static, 1)
  for (std::size_t i = 0; i < m_size; ++i) {
    data_ptr[i] = static_cast<unsigned char>(i);
  }
}
#endif

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
  double m_size = 42;
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

TEST_P(AllocatorAccessibilityTest, AccessibilityFromUndefined)
{
  ::testing::FLAGS_gtest_death_test_style = "threadsafe"; 
  if(is_accessible(cPlatform::undefined, *m_allocator)) {
    FAIL() << "An Undefined platform is not accessible." << std::endl;
  } else {
    SUCCEED(); //Succeed every time we can't access an undefined platform.
  }
}

#if defined(UMPIRE_ENABLE_CUDA)
TEST_P(AllocatorAccessibilityTest, AccessibilityFromCuda)
{
  ::testing::FLAGS_gtest_death_test_style = "threadsafe"; 
  if(is_accessible(cPlatform::cuda, *m_allocator)) {
    double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
    ASSERT_NO_THROW(test(data, m_size));
  } else {
    if(m_allocator->getAllocationStrategy()->getTraits().resource == myResource::FILE)
      SUCCEED(); //FILE should not be accessed from CUDA
    else {
      double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
      ASSERT_DEATH(test(data, m_size), "");
    }
  }
}
#endif

#if defined(UMPIRE_ENABLE_HIP)
TEST_P(AllocatorAccessibilityTest, AccessibilityFromHip)
{
  ::testing::FLAGS_gtest_death_test_style = "threadsafe"; 
  if(is_accessible(cPlatform::hip, *m_allocator)) {
    double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
    ASSERT_NO_THROW(test(data, m_size));
  } else {
    double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
    ASSERT_DEATH(test(data, m_size), "");
  }
}
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
TEST_P(AllocatorAccessibilityTest, AccessibilityFromOpenMP)
{
  ::testing::FLAGS_gtest_death_test_style = "threadsafe"; 
  if(is_accessible(cPlatform::omp_target, *m_allocator)) {
    double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
    ASSERT_NO_THROW(test(data, m_size));
  } else {
    double* data = static_cast<double*>(m_allocator->allocate(m_size * sizeof(double)));
    ASSERT_DEATH(test(data, m_size), "");
  }
}
#endif

/////////////////////////////
//Sycl test not yet available
/////////////////////////////

std::vector<std::string> get_allocators()
{
  auto& rm = umpire::ResourceManager::getInstance();
  std::vector<std::string> available_allocators = rm.getResourceNames();
  return available_allocators;
}

INSTANTIATE_TEST_SUITE_P(Allocators, AllocatorAccessibilityTest, ::testing::ValuesIn(get_allocators()));

//END gtest
