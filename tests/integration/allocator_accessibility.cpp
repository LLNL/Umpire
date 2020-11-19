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
#include "umpire/util/MemoryResourceTraits.hpp"
#include "umpire/strategy/QuickPool.hpp"

struct host_platform {};

template <typename Platform>
struct allocate_and_use{};

template<>
struct allocate_and_use<host_platform>
{
  void test(umpire::Allocator* alloc, size_t size)
  {
    size_t* data = static_cast<size_t*>(alloc->allocate(size * sizeof(size_t)));
    data[0] = size * size;
    alloc->deallocate(data);
  }
};

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
__global__ void tester(size_t* d_data, size_t size)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
   
  if (idx == 0) {
    d_data[0] = size * size;
  }
}
#endif

#if defined(UMPIRE_ENABLE_CUDA)
struct cuda_platform {};

template<>
struct allocate_and_use<cuda_platform>
{
  void test(umpire::Allocator* alloc, size_t size)
  {
    size_t* data = static_cast<size_t*>(alloc->allocate(size * sizeof(size_t)));
    tester<<<1, 16>>>(data, size);
    cudaDeviceSynchronize();
    alloc->deallocate(data);
  }
};
#endif

#if defined(UMPIRE_ENABLE_HIP)
struct hip_platform{};

template<>
struct allocate_and_use<hip_platform>
{
  void test(umpire::Allocator* alloc, size_t size)
  {
    size_t* data = static_cast<size_t*>(alloc->allocate(size * sizeof(size_t)));
    hipLaunchKernelGGL(tester, dim3(1), dim3(16), 0,0, data, size);
    hipDeviceSynchronize();
    alloc->deallocate(data);
  }
};
#endif

#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
struct omp_target_platform{};

template<>
struct allocate_and_use<omp_target_platform>
{
  void test(umpire::Allocator* alloc, size_t size)
  {
    int dev = alloc->getAllocationStrategy()->getTraits().id; 
    size_t* data = static_cast<size_t*>(alloc->allocate(size * sizeof(size_t)));
    size_t* d_data{static_cast<size_t*>(data)};

#pragma omp target is_device_ptr(d_data) device(dev)
#pragma omp teams distribute parallel for schedule(static, 1)
    for (auto i = 0; i < size; ++i) {
      d_data[i] = static_cast<size_t>(i);
    }

    alloc->deallocate(data);
  }
};
#endif

class AllocatorAccessibilityTest : public ::testing::TestWithParam<std::string> {
 public:
  virtual void SetUp()
  {
    auto& rm = umpire::ResourceManager::getInstance();
    m_allocator = new umpire::Allocator(rm.getAllocator(GetParam()));
    m_allocator_pool = new umpire::Allocator(rm.makeAllocator<umpire::strategy::QuickPool>
        ("pool_" + GetParam(), *m_allocator, 42 * sizeof(size_t), 1));
  }

  virtual void TearDown()
  {
    m_allocator->release();
    m_allocator_pool->release();
    if(m_allocator)
      delete m_allocator;
    if(m_allocator_pool)
      delete m_allocator_pool;
  }
  
  umpire::Allocator* m_allocator;
  umpire::Allocator* m_allocator_pool;
  size_t m_size = 42;
};

void run_access_test(umpire::Allocator* alloc, size_t size)
{::testing::FLAGS_gtest_death_test_style = "threadsafe";

  if(umpire::is_accessible(umpire::Platform::host, *alloc)) {
    allocate_and_use<host_platform> host;
    ASSERT_NO_THROW(host.test(alloc, size));
  }
  else {
    allocate_and_use<host_platform> host;
    ASSERT_DEATH(host.test(alloc, size), "");
  }

#if defined(UMPIRE_ENABLE_CUDA)
  if (umpire::is_accessible(umpire::Platform::cuda, *alloc)) {
    allocate_and_use<cuda_platform> cuda;
    ASSERT_NO_THROW(cuda.test(alloc, size));
  }
  else if (alloc->getAllocationStrategy()->getTraits().resource
           == umpire::MemoryResourceTraits::resource_type::file)
    SUCCEED();
  else {
    allocate_and_use<cuda_platform> cuda;
    ASSERT_DEATH(cuda.test(alloc, size), "");
  }
#endif
  
#if defined(UMPIRE_ENABLE_HIP)
  if (umpire::is_accessible(umpire::Platform::hip, *alloc)) {
    allocate_and_use<hip_platform> hip;
    ASSERT_NO_THROW(hip.test(alloc, size));
  }
  else {
    allocate_and_use<hip_platform> hip;
    ASSERT_DEATH(hip.test(alloc, size), "");
  }
#endif
 
#if defined(UMPIRE_ENABLE_OPENMP_TARGET)
  if (umpire::is_accessible(umpire::Platform::omp_target, *alloc)) {
    allocate_and_use<omp_target_platform> omp;
    ASSERT_NO_THROW(omp.test(alloc, size));
  }
  else {
    allocate_and_use<omp_target_platform> omp;
    ASSERT_DEATH(omp.test(alloc, size), "");
  }
#endif

/////////////////////////////
//Sycl test not yet available
/////////////////////////////

  if(umpire::is_accessible(umpire::Platform::undefined, *alloc)) {
    FAIL() << "An Undefined platform is not accessible." << std::endl;
  }
  else
    SUCCEED();
}

TEST_P(AllocatorAccessibilityTest, AllocatorAccessibilityFromPlatform)
{
  run_access_test(m_allocator, m_size);
  run_access_test(m_allocator_pool, m_size);
}

std::vector<std::string> get_allocators()
{
  auto& rm = umpire::ResourceManager::getInstance();
  std::vector<std::string> all_allocators = rm.getResourceNames();
  std::vector<std::string> avail_allocators;
  
  std::cout << "Available allocators: ";
  for(auto a : all_allocators) {
    if(a.find("::") == std::string::npos) { 
      avail_allocators.push_back(a);
      std::cout << a << " ";
    }
  }
  std::cout << std::endl;

  return avail_allocators;
}

INSTANTIATE_TEST_SUITE_P(Allocators, AllocatorAccessibilityTest, ::testing::ValuesIn(get_allocators()));

//END gtest
