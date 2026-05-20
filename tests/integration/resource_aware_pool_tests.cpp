/////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
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
#include "umpire/util/wrap_allocator.hpp"

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

  // EXPECT_THROW(m_pool.deallocate(ptr, r2), umpire::runtime_error);
  EXPECT_NO_THROW(m_pool.deallocate(ptr, r1));

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
  // EXPECT_EQ(ptr, ptr2); // multiple device resources, but with explicit sync, ptr is same
}

TEST_P(ResourceAwarePoolTest, ReleaseCheck)
{
  resource_type d1;

  double* ptr = static_cast<double*>(m_pool.allocate(1024, d1));
  EXPECT_EQ(get_resource(m_pool, ptr), Resource{d1});

  do_sleep<<<1, 32, 0, d1.get_stream()>>>(ptr);

  m_pool.deallocate(ptr, d1);
  EXPECT_EQ(get_num_pending(m_pool), 1);

  EXPECT_NO_THROW(m_pool.release());
}

TEST_P(ResourceAwarePoolTest, CoalescingAdjacentChunks)
{
  resource_type d1;
  
  // Allocate two adjacent chunks
  void* ptr1 = m_pool.allocate(1024, d1);
  void* ptr2 = m_pool.allocate(1024, d1);
  void* ptr3 = m_pool.allocate(1024, d1);
  
  // Deallocate non-adjacent chunks
  m_pool.deallocate(ptr1, d1);
  m_pool.deallocate(ptr3, d1);
  
  // Wait for any pending ops
  d1.get_event().wait();
  
  // Deallocate middle chunk - should trigger coalescing
  m_pool.deallocate(ptr2, d1);
  
  // Force coalesce
  m_pool.release();
  
  // Should be able to allocate a larger chunk now
  void* large_ptr = m_pool.allocate(3072, d1);
  EXPECT_NE(large_ptr, nullptr);
  
  m_pool.deallocate(large_ptr, d1);
}

TEST_P(ResourceAwarePoolTest, PendingToFreeTransition)
{
  resource_type d1, d2;
  
  double* ptr = static_cast<double*>(m_pool.allocate(2048, d1));
  
  // Start async operation
  do_sleep<<<1, 32, 0, d1.get_stream()>>>(ptr);
  
  // Deallocate - goes to pending
  m_pool.deallocate(ptr, d1);
  EXPECT_EQ(get_num_pending(m_pool), 1);
  
  // Try to allocate same size with same resource while pending
  double* ptr2 = static_cast<double*>(m_pool.allocate(2048, d1));
  
  // Should reuse the pending chunk if it's done, or get new chunk
  EXPECT_EQ(get_num_pending(m_pool), 0); // pending chunk was reused
  
  m_pool.deallocate(ptr2, d1);
}

TEST_P(ResourceAwarePoolTest, ResourceMismatchOnDeallocate)
{
  resource_type d1, d2;
  
  double* ptr = static_cast<double*>(m_pool.allocate(1024, d1));
  
  // Correct deallocation should work
  EXPECT_NO_THROW(m_pool.deallocate(ptr, d1));

  double* ptr2 = static_cast<double*>(m_pool.allocate(2048, d1));
  
  // Try to deallocate with wrong resource
  EXPECT_THROW(m_pool.deallocate(ptr2, d2), umpire::runtime_error);
  
}

TEST_P(ResourceAwarePoolTest, MultiplePendingChunksSameResource)
{
  resource_type d1;
  Resource r1{d1};
  
  // Create multiple allocations
  std::vector<double*> ptrs;
  for (int i = 0; i < 5; ++i) {
    ptrs.push_back(static_cast<double*>(m_pool.allocate(1024, r1)));
    do_sleep<<<1, 32, 0, d1.get_stream()>>>(ptrs[i]);
  }
  
  // Deallocate all - should create multiple pending entries
  for (auto ptr : ptrs) {
    m_pool.deallocate(ptr, r1);
  }
  
  EXPECT_EQ(get_num_pending(m_pool), 5);
  
  // Allocate with same resource - should reuse pending
  double* new_ptr = static_cast<double*>(m_pool.allocate(1024, r1));
  EXPECT_LT(get_num_pending(m_pool), 5); // One was reused
  
  m_pool.deallocate(new_ptr, r1);
}

TEST_P(ResourceAwarePoolTest, DestructorWithPendingChunks)
{
  auto& rm = umpire::ResourceManager::getInstance();
  {
    auto pool = rm.makeAllocator<umpire::strategy::ResourceAwarePool>(
        "temp-pool" + GetParam(), rm.getAllocator(GetParam()));
    
    resource_type d1;
    double* ptr = static_cast<double*>(pool.allocate(1024, d1));
    
    do_sleep<<<1, 32, 0, d1.get_stream()>>>(ptr);
    
    pool.deallocate(ptr, d1);
    EXPECT_EQ(get_num_pending(pool), 1);
    
    // Pool destructor should handle pending chunks correctly
  } // Pool destroyed here
  
  // Should not crash or leak
}

TEST_P(ResourceAwarePoolTest, ChunkSplittingAndMerging)
{
  resource_type d1;
  
  // Allocate large chunk
  void* large = m_pool.allocate(4096, d1);
  m_pool.deallocate(large, d1);
  d1.get_event().wait();
  
  // Allocate smaller chunk - should split
  void* small1 = m_pool.allocate(1024, d1);
  void* small2 = m_pool.allocate(1024, d1);
  
  // Deallocate in order that promotes merging
  m_pool.deallocate(small1, d1);
  d1.get_event().wait();
  m_pool.deallocate(small2, d1);
  d1.get_event().wait();
  
  // Should be able to allocate large chunk again
  void* large2 = m_pool.allocate(4096, d1);
  EXPECT_NE(large2, nullptr);
  
  m_pool.deallocate(large2, d1);
}

TEST_P(ResourceAwarePoolTest, GetResourceEdgeCases)
{
  resource_type d1;
  
  double* ptr = static_cast<double*>(m_pool.allocate(1024, d1));
  
  // Valid pointer in used map
  EXPECT_EQ(get_resource(m_pool, ptr), Resource{d1});
  
  // After deallocation - goes to pending
  do_sleep<<<1, 32, 0, d1.get_stream()>>>(ptr);
  m_pool.deallocate(ptr, d1);
  
  // Should still find it in pending
  EXPECT_EQ(get_num_pending(m_pool), 1);
  EXPECT_EQ(get_resource(m_pool, ptr), Resource{d1});
  
  // Wait for completion
  d1.get_event().wait();

  // Force processing of pending 
  // (release only processes pending chunks upon destruction, so call coalesce instead)
  auto rap = umpire::util::unwrap_allocator<umpire::strategy::ResourceAwarePool>(m_pool);
  rap->coalesce();

  EXPECT_EQ(get_num_pending(m_pool), 0);
  
  // Now it's free - should return warning and default
  camp::resources::Resource res = get_resource(m_pool, ptr);
  EXPECT_TRUE(res == camp::resources::Resource{Host{}});
  
  // Invalid pointer
  double* invalid_ptr = reinterpret_cast<double*>(0xDEADBEEF);
  res = get_resource(m_pool, invalid_ptr);
  EXPECT_EQ(res, camp::resources::Resource{Host{}});
}

TEST_P(ResourceAwarePoolTest, StressTestMultipleResources)
{
  const int num_iterations = 100;
  std::vector<resource_type> resources(4);
  std::vector<void*> allocations;
  std::vector<Resource> allocation_resources;
  
  for (int i = 0; i < num_iterations; ++i) {
    int res_idx = i % resources.size();
    resource_type& res = resources[res_idx];
    
    void* ptr = m_pool.allocate(512 + (i * 128) % 2048, res);
    allocations.push_back(ptr);
    allocation_resources.push_back(Resource{res});
    
    // Deallocate some
    if (i > 10 && i % 3 == 0) {
      int idx = (i - 10) / 2;
      m_pool.deallocate(allocations[idx], allocation_resources[idx]);
      allocations[idx] = nullptr;
    }
  }
  
  // Clean up remaining
  for (size_t i = 0; i < allocations.size(); ++i) {
    if (allocations[i]) {
      m_pool.deallocate(allocations[i], allocation_resources[i]);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ResourceAwarePoolTests, ResourceAwarePoolTest, ::testing::ValuesIn(get_allocator_strings()));

#endif
