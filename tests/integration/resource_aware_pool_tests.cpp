//////////////////////////////////////////////////////////////////////////////
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
#include "umpire/config.hpp"
#include "umpire/strategy/ResourceAwarePool.hpp"
#include "umpire/Umpire.hpp"

using clock_value_t = long long;
using namespace camp::resources;

__device__ void my_sleep(clock_value_t sleep_cycles)
{
  clock_value_t start = clock64();
  clock_value_t cycles_elapsed;
  do {
    cycles_elapsed = clock64() - start;
  } while (cycles_elapsed < sleep_cycles);
}

__global__ void do_sleep(double* ptr)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  ptr[id] = id * 1.0;
  my_sleep(1000000);
  ptr[id] = id * 2.0;
}

TEST(ResourceAwarePoolTest, Construction)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool", rm.getAllocator("HOST"));

  EXPECT_EQ(pool.getCurrentSize(), 0);
}

TEST(ResourceAwarePoolTest, Check_States)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>("rap-pool2", rm.getAllocator("DEVICE"));

#if defined(UMPIRE_ENABLE_CUDA)
  Cuda d1, d2;
#elif defined(UMPIRE_ENABLE_HIP)
  Hip d1, d2;
#else
  Host d1, d2;
#endif

  Resource r1{d1}, r2{d2};

  double* ptr = static_cast<double*>(pool.allocate(r1, 1024));
  double* compare_ptr1 = ptr;
  
  EXPECT_EQ(getResource(pool, ptr), r1);
  EXPECT_EQ(getPendingSize(pool), 0);

#if defined(UMPIRE_ENABLE_CUDA)
  do_sleep<<<1, 32, 0, d1.get_stream()>>>(ptr);
#elif defined(UMPIRE_ENABLE_HIP)
  hipLaunchKernelGGL(do_sleep, 1, 32, 0, d1.get_stream(), ptr);
#else
#endif

  pool.deallocate(ptr);
  EXPECT_EQ(getPendingSize(pool), 1);
  
  ptr = static_cast<double*>(pool.allocate(r2, 2048));
  double* compare_ptr2 = ptr;
 
#if defined(UMPIRE_ENABLE_DEVICE)
  EXPECT_NE(compare_ptr1, compare_ptr2); 
#else
  EXPECT_EQ(compare_ptr1, compare_ptr2); 
#endif
}
