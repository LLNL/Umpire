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

#if defined(UMPIRE_ENABLE_CUDA)
using resource_type = Cuda;
#elif defined(UMPIRE_ENABLE_HIP)
using resource_type = Hip;
#endif

std::string unique_name()
{
  static int unique_name_id{0};
  std::stringstream ss;

  ss << "_Unique_Name_" << unique_name_id++;
  return ss.str();
}

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

TEST(ResourceAwarePool_Host_Test, Check_States_Host)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool-host", rm.getAllocator("HOST"));

  Resource r1{Host{}}, r2{Host{}};
  int* ptr = static_cast<int*>(pool.allocate(1024, r1));
  int* compare_ptr1 = ptr;

  EXPECT_EQ(get_resource(pool, ptr), r1);
  EXPECT_EQ(get_num_pending(pool), 0);

  host_sleep(ptr);

  pool.deallocate(ptr, r1);
  EXPECT_EQ(get_num_pending(pool), 0); // When only using host, there will be no pending chunks

  ptr = static_cast<int*>(pool.allocate(1024, r2));
  int* compare_ptr2 = ptr;

  EXPECT_TRUE(r1 == r2);
  EXPECT_EQ(compare_ptr1, compare_ptr2); // only 1 host resource available, no possible data race
  pool.deallocate(ptr, r2);
}

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)

using clock_value_t = long long;

__device__ clock_value_t my_clock()
{
  return clock64();
}

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
    m_pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>(
        std::string{"rap-pool-" + GetParam() + unique_name()}, rm.getAllocator(GetParam()));
  }

  virtual void TearDown()
  {
    m_pool.release();
  }

  umpire::Allocator m_pool;
};

TEST_P(ResourceAwarePoolTest, CheckStates)
{
  resource_type d1, d2;
  Resource r1{d1}, r2{d2};

  double* ptr = static_cast<double*>(m_pool.allocate(1024, r1));

  EXPECT_EQ(get_resource(m_pool, ptr), r1);
  EXPECT_EQ(get_num_pending(m_pool), 0);

  do_sleep<<<1, 32, 0, d1.get_stream()>>>(ptr);

  m_pool.deallocate(ptr, r1);

  EXPECT_EQ(get_num_pending(m_pool), 1);

  double* ptr2 = static_cast<double*>(m_pool.allocate(1024, r2));

  EXPECT_FALSE(r1 == r2);
  EXPECT_EQ(get_resource(m_pool, ptr2), r2);
  EXPECT_NE(ptr, ptr2); // multiple device resources, possible data race, needs different addr
}

TEST_P(ResourceAwarePoolTest, ExplicitSync)
{
  resource_type d1, d2;

  double* ptr = static_cast<double*>(m_pool.allocate(1024, d1));
  EXPECT_EQ(get_resource(m_pool, ptr), Resource{d1});

  do_sleep<<<1, 32, 0, d1.get_stream()>>>(ptr);

  m_pool.deallocate(ptr, d1);
  d1.get_event().wait(); // explicitly sync the device streams (camp resources)
  double* ptr2 = static_cast<double*>(m_pool.allocate(1024, d2));

  EXPECT_EQ(get_resource(m_pool, ptr2), Resource{d2});
  EXPECT_FALSE(d1 == d2);
  EXPECT_EQ(ptr, ptr2); // multiple device resources, but with explicit sync, ptr is same
}

TEST_P(ResourceAwarePoolTest, ReleaseCheck)
{
  resource_type d1;

  double* ptr = static_cast<double*>(m_pool.allocate(1024, d1));
  EXPECT_EQ(get_resource(m_pool, ptr), Resource{d1});

  do_sleep<<<1, 32, 0, d1.get_stream()>>>(ptr);

  m_pool.deallocate(ptr, d1);
  EXPECT_EQ(get_num_pending(m_pool), 1);

  m_pool.release();
  EXPECT_EQ(get_num_pending(m_pool), 0);
}

INSTANTIATE_TEST_SUITE_P(ResourceAwarePoolTests, ResourceAwarePoolTest, ::testing::ValuesIn(get_allocator_strings()));

#endif
