/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <stdio.h>

#include <iostream>

#include "camp/camp.hpp"
#include "gtest/gtest.h"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/ResourceAwarePool.hpp"

using namespace camp::resources;

void host_sleep(int* ptr)
{
  int i = 0;
  while (i < 1000000) {
    int y = i;
    y++;
    i = y;
  }
  *ptr = i;
  ptr++;
}

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)

using clock_value_t = long long;

#if defined(UMPIRE_ENABLE_CUDA)
__device__ clock_value_t my_clock()
{
  return clock64();
}
#elif defined(UMPIRE_ENABLE_HIP)
__device__ clock_value_t my_clock()
{
  return hipGetClock();
}
#endif

__device__ void my_sleep(clock_value_t sleep_cycles)
{
  clock_value_t start = my_clock();
  clock_value_t cycles_elapsed;
  do {
    cycles_elapsed = my_clock() - start;
  } while (cycles_elapsed < sleep_cycles);
}

__global__ void do_sleep(double* ptr)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  ptr[id] = id * 1.0;
  my_sleep(1000000);
  ptr[id] = id * 2.0;
}

std::vector<std::string> get_allocator_strings()
{
  std::vector<std::string> allocators;

  allocators.push_back("DEVICE");
  // auto& rm = umpire::ResourceManager::getInstance();
  // for (int id = 0; id < rm.getNumDevices(); id++) {
  // allocators.push_back(std::string{"DEVICE::" + std::to_string(id)});
  //}
#if defined(UMPIRE_ENABLE_UM)
  allocators.push_back("UM");
#endif
#if defined(UMPIRE_ENABLE_CONST)
  allocators.push_back("DEVICE_CONST");
#endif
#if defined(UMPIRE_ENABLE_PINNED)
  allocators.push_back("PINNED");
#endif

  return allocators;
}

class ResourceAwarePoolTest : public ::testing::TestWithParam<std::string> {
 public:
  virtual void SetUp()
  {
    auto& rm = umpire::ResourceManager::getInstance();
    m_pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>(std::string{"rap-pool-" + GetParam()},
                                                                   rm.getAllocator(GetParam()));
  }

  virtual void TearDown()
  {
    m_pool.release();
  }

  umpire::Allocator m_pool;
};

TEST_P(ResourceAwarePoolTest, Check_States)
{
#if defined(UMPIRE_ENABLE_CUDA)
  Cuda d1, d2;
#elif defined(UMPIRE_ENABLE_HIP)
  Hip d1, d2;
#endif

  Resource r1{d1}, r2{d2};

  double* ptr = static_cast<double*>(m_pool.allocate(r1, 1024));

  EXPECT_EQ(getResource(m_pool, ptr), r1);
  EXPECT_EQ(getPendingSize(m_pool), 0);

#if defined(UMPIRE_ENABLE_CUDA)
  do_sleep<<<1, 32, 0, d1.get_stream()>>>(ptr);
#elif defined(UMPIRE_ENABLE_HIP)
  hipLaunchKernelGGL(do_sleep, 1, 32, 0, d1.get_stream(), ptr);
#endif

  m_pool.deallocate(ptr);
  EXPECT_EQ(getPendingSize(m_pool), 1);

  double* ptr2 = static_cast<double*>(m_pool.allocate(r2, 2048));

  EXPECT_FALSE(r1 == r2);
  EXPECT_EQ(getResource(m_pool, ptr2), r2);
  EXPECT_NE(ptr, ptr2); // multiple device resources, possible data race, needs different addr
}

INSTANTIATE_TEST_SUITE_P(ResourceAwarePoolTests, ResourceAwarePoolTest, ::testing::ValuesIn(get_allocator_strings()));

#endif

TEST(ResourceAwarePool_Host_Test, Check_States_Host)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool-host", rm.getAllocator("HOST"));

  Host h1, h2;
  Resource r1{h1}, r2{h2};

  int* ptr = static_cast<int*>(pool.allocate(r1, 1024));
  int* compare_ptr1 = ptr;

  EXPECT_EQ(getResource(pool, ptr), r1);
  EXPECT_EQ(getPendingSize(pool), 0);

  host_sleep(ptr);

  pool.deallocate(ptr);
  EXPECT_EQ(getPendingSize(pool), 0); // When only using host, there will be no pending chunks

  ptr = static_cast<int*>(pool.allocate(r2, 2048));
  int* compare_ptr2 = ptr;

  EXPECT_TRUE(r1 == r2);
  EXPECT_EQ(compare_ptr1, compare_ptr2); // only 1 host resource available, no possible data race
}
