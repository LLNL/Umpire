//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#include <string>
#include <sstream>

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/DynamicPoolHeuristic.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"
#include "umpire/strategy/FixedPool.hpp"
#include "umpire/strategy/MixedPool.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/SizeLimiter.hpp"

#if defined(UMPIRE_ENABLE_NUMA)
#include "umpire/strategy/NumaPolicy.hpp"
#include "umpire/util/numa.hpp"
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

static int unique_pool_name = 0;
static int initial_min_size = 1024;
static int subsequent_min_size = 512;

const char* AllocationDevices[] = {
  "HOST"
#if defined(UMPIRE_ENABLE_DEVICE)
    , "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_UM)
    , "UM"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
    , "PINNED"
#endif
};

class StrategyTest :
  public ::testing::TestWithParam<const char*>
{
public:
  void SetUp() override {
    auto& rm = umpire::ResourceManager::getInstance();
    allocatorName = GetParam();
    poolName << allocatorName << "_pool_" << unique_pool_name++;

    rm.makeAllocator<umpire::strategy::DynamicPool>
                  (  poolName.str()
                   , rm.getAllocator(allocatorName)
                   , initial_min_size
                   , subsequent_min_size);

    allocator = new umpire::Allocator(rm.getAllocator(poolName.str()));
  }

  void TearDown() override {
    delete allocator;
    allocator = nullptr;
  }

  umpire::Allocator* allocator;
  std::string allocatorName;
  std::stringstream poolName;
};

TEST_P(StrategyTest, Allocate) {
  void* alloc = nullptr;
  alloc = allocator->allocate(100);
  allocator->deallocate(alloc);
}

TEST_P(StrategyTest, Sizes) {
  void* alloc = nullptr;
  ASSERT_NO_THROW({ alloc = allocator->allocate(100); });
  ASSERT_EQ(allocator->getSize(alloc), 100);
  ASSERT_GE(allocator->getCurrentSize(), 100);
  ASSERT_EQ(allocator->getHighWatermark(), 100);
  ASSERT_GE(allocator->getActualSize(), initial_min_size);

  void* alloc2 = nullptr;
  ASSERT_NO_THROW({ alloc2 = allocator->allocate(initial_min_size); });
  ASSERT_NO_THROW({ allocator->deallocate(alloc); });

  ASSERT_GE(allocator->getCurrentSize(), initial_min_size);
  ASSERT_EQ(allocator->getHighWatermark(), initial_min_size+100);
  ASSERT_GE(allocator->getActualSize(), initial_min_size+subsequent_min_size);
  ASSERT_EQ(allocator->getSize(alloc2), initial_min_size);

  ASSERT_NO_THROW({ allocator->deallocate(alloc2); });
}

TEST_P(StrategyTest, Duplicate)
{
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_TRUE(rm.isAllocator(allocatorName));

  ASSERT_EQ(allocator->getName(), poolName.str());

  ASSERT_ANY_THROW(
      rm.makeAllocator<umpire::strategy::DynamicPool>(
        poolName.str(), rm.getAllocator(allocatorName)));
}

INSTANTIATE_TEST_CASE_P(Allocations, StrategyTest, ::testing::ValuesIn(AllocationDevices),);

#if defined(UMPIRE_ENABLE_DEVICE)
TEST(SimpoolStrategy, Device)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.getAllocator("DEVICE");
  void* alloc = nullptr;

  // Determine how much memory we can allocate from device
  std::size_t max_mem = 0;
  const std::size_t OneGiB = 1 * 1024 * 1024 * 1024;
  try {
    while ( true ) {  // Will "catch" out when allocation fails
      alloc = allocator.allocate(max_mem + OneGiB);
      ASSERT_NO_THROW( { allocator.deallocate(alloc); } );
      max_mem += OneGiB;
    }
  }
  catch (...) {
    ASSERT_GT(max_mem, OneGiB);
  }

  allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "device_simpool", rm.getAllocator("DEVICE"));

  ASSERT_EQ(allocator.getName(), "device_simpool");

  ASSERT_NO_THROW( { alloc = allocator.allocate(100); } );
  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_NO_THROW( { allocator.deallocate(alloc); } );

  std::size_t alloc_size = max_mem / 4;
  void* alloc1 = nullptr;
  void* alloc2 = nullptr;
  void* alloc3 = nullptr;

  // Hold a little of the first block we allocate
  ASSERT_NO_THROW( { alloc1 = allocator.allocate(1024); } );
  ASSERT_NO_THROW( { alloc2 = allocator.allocate(1024); } );
  ASSERT_NO_THROW( { allocator.deallocate(alloc1); } );
  ASSERT_NO_THROW( { alloc3 = allocator.allocate(100); } );
  ASSERT_NO_THROW( { allocator.deallocate(alloc2); } );

  for (int i = 0; i < 16; ++i) {
    ASSERT_NO_THROW( { alloc1 = allocator.allocate(alloc_size); } );
    ASSERT_NO_THROW( { allocator.deallocate(alloc1); } );
    alloc_size += 1024*1024;
  }

  ASSERT_NO_THROW( { allocator.deallocate(alloc3); } );
}
#endif

TEST(MonotonicStrategy, Host)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
      "host_monotonic_pool", 65536, rm.getAllocator("HOST"));

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "host_monotonic_pool");

  allocator.deallocate(alloc);
}

#if defined(UMPIRE_ENABLE_DEVICE)
TEST(MonotonicStrategy, Device)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
      "device_monotonic_pool", 65536, rm.getAllocator("DEVICE"));

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "device_monotonic_pool");

  allocator.deallocate(alloc);
}
#endif // defined(UMPIRE_ENABLE_DEVICE)

#if defined(UMPIRE_ENABLE_UM)
TEST(MonotonicStrategy, UM)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
      "um_monotonic_pool", 65536, rm.getAllocator("UM"));

  void* alloc = allocator.allocate(100);

  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "um_monotonic_pool");

  allocator.deallocate(alloc);
}
#endif // defined(UMPIRE_ENABLE_UM)

#if defined(UMPIRE_ENABLE_CUDA)
TEST(AllocationAdvisor, Create)
{
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_NO_THROW(
    auto read_only_alloc =
      rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
        "read_only_um", rm.getAllocator("UM"), "READ_MOSTLY");
    UMPIRE_USE_VAR(read_only_alloc));

  ASSERT_ANY_THROW(
    auto failed_alloc =
      rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
          "read_only_um_nonsense_operator", rm.getAllocator("UM"), "FOOBAR");
    UMPIRE_USE_VAR(failed_alloc));
}

TEST(AllocationAdvisor, CreateWithId)
{
  auto& rm = umpire::ResourceManager::getInstance();

  const int device_id = 2;

  ASSERT_NO_THROW(
    auto read_only_alloc =
    rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
      "read_only_um_device_id", rm.getAllocator("UM"), "READ_MOSTLY", device_id);
    UMPIRE_USE_VAR(read_only_alloc));

  ASSERT_ANY_THROW(
    auto failed_alloc =
      rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
        "read_only_um_nonsense_operator_device_id",
      rm.getAllocator("UM"), "FOOBAR", device_id);
    UMPIRE_USE_VAR(failed_alloc));
}

TEST(AllocationAdvisor, Host)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto um_allocator = rm.getAllocator("UM");
  auto host_allocator = rm.getAllocator("HOST");

  auto read_only_alloc =
    rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
      "preferred_location_host", um_allocator, "PREFERRED_LOCATION", host_allocator);

  ASSERT_NO_THROW({
      double* data = static_cast<double*>(
          read_only_alloc.allocate(1024*sizeof(double)));
      read_only_alloc.deallocate(data);
  });

}
#endif // defined(UMPIRE_ENABLE_CUDA)

TEST(FixedPool, Host)
{
  auto& rm = umpire::ResourceManager::getInstance();

  const int data_size = 100 * sizeof(int);

  auto allocator = rm.makeAllocator<umpire::strategy::FixedPool>(
    "host_fixed_pool", rm.getAllocator("HOST"), data_size, 64);

  void* alloc = allocator.allocate(data_size);

  ASSERT_EQ(allocator.getCurrentSize(), data_size);
  ASSERT_GE(allocator.getActualSize(), data_size*64);
  ASSERT_EQ(allocator.getSize(alloc), data_size);
  ASSERT_GE(allocator.getHighWatermark(), data_size);
  ASSERT_EQ(allocator.getName(), "host_fixed_pool");

  allocator.deallocate(alloc);
}

TEST(MixedPool, Host)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::MixedPool>(
      "host_mixed_pool", rm.getAllocator("HOST"));

  const size_t max_power = 9;
  void* alloc[max_power];
  size_t size = 4, total_size = 0;
  for (size_t i = 0; i < max_power; ++i) {
    alloc[i] = allocator.allocate(size);
    total_size += size;
    size *= 4;
  }

  ASSERT_EQ(allocator.getCurrentSize(), total_size);
  ASSERT_GT(allocator.getActualSize(), total_size);
  ASSERT_EQ(allocator.getSize(alloc[max_power-1]), size/4);
  ASSERT_GE(allocator.getHighWatermark(), total_size);
  ASSERT_EQ(allocator.getName(), "host_mixed_pool");

  for (size_t i = 0; i < max_power; ++i)
    allocator.deallocate(alloc[i]);
}

#if defined(_OPENMP)
TEST(ThreadSafeAllocator, Host)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      "thread_safe_allocator", rm.getAllocator("HOST"));

#pragma omp parallel
  {
    const size_t size = 1024*omp_get_thread_num();

    double* thread_data = static_cast<double*>(
     allocator.allocate(size*sizeof(double)));

    allocator.deallocate(thread_data);
  }

  SUCCEED();
}

#endif

TEST(SizeLimiter, Host)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.makeAllocator<umpire::strategy::SizeLimiter>(
      "size_limited_alloc", rm.getAllocator("HOST"), 64);

  void* data = nullptr;

  EXPECT_NO_THROW(data = alloc.allocate(64));

  EXPECT_THROW({
    void* tmp_data = alloc.allocate(1024);
    UMPIRE_USE_VAR(tmp_data);
  }, umpire::util::Exception);

  EXPECT_NO_THROW(
    alloc.deallocate(data));
}

TEST(ReleaseTest, Works)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "host_simpool_for_release", rm.getAllocator("HOST"), 64, 64);

  void* ptr_one = alloc.allocate(62);
  void* ptr_two = alloc.allocate(1024);
  alloc.deallocate(ptr_two);

  EXPECT_NO_THROW(alloc.release());

  alloc.deallocate(ptr_one);
}

TEST(HeuristicTest, OutOfBounds)
{
  EXPECT_THROW({
    auto h = umpire::strategy::heuristic_percent_releasable(-1);
    UMPIRE_USE_VAR(h);
  }, umpire::util::Exception);

  EXPECT_THROW({
    auto h = umpire::strategy::heuristic_percent_releasable(101);
    UMPIRE_USE_VAR(h);
  }, umpire::util::Exception);
}

TEST(HeuristicTest, EdgeCases_75)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto h_fun = umpire::strategy::heuristic_percent_releasable(75);

  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "host_dyn_pool_h_75", rm.getAllocator("HOST"),
      1024ul, 1024ul, h_fun);

  auto strategy = alloc.getAllocationStrategy();
  auto tracker = dynamic_cast<umpire::strategy::AllocationTracker*>(strategy);

  if (tracker) {
    strategy = tracker->getAllocationStrategy();
  }

  auto dynamic_pool = dynamic_cast<umpire::strategy::DynamicPool*>(strategy);

  ASSERT_NE(dynamic_pool, nullptr);

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 0);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  void* a[4];
  for ( int i = 0; i < 4; ++i ) {
    ASSERT_NO_THROW({ a[i] = alloc.allocate(1024); });
    ASSERT_EQ(alloc.getActualSize(), (1024*(i+1)));
    ASSERT_EQ(dynamic_pool->getBlocksInPool(), (i+1));
    ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);
  }

  for ( int i = 0; i < 2; ++i ) {
    ASSERT_NO_THROW({ alloc.deallocate(a[i]); });
    ASSERT_EQ(dynamic_pool->getBlocksInPool(), 4);
  }

  ASSERT_NO_THROW({ alloc.deallocate(a[2]); });
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);

  ASSERT_NO_THROW({ alloc.deallocate(a[3]); });
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
}

TEST(HeuristicTest, EdgeCases_100)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto h_fun = umpire::strategy::heuristic_percent_releasable(100);

  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "host_dyn_pool_h_100", rm.getAllocator("HOST"),
      initial_min_size, subsequent_min_size, h_fun);

  auto strategy = alloc.getAllocationStrategy();
  auto tracker = dynamic_cast<umpire::strategy::AllocationTracker*>(strategy);

  if (tracker) {
    strategy = tracker->getAllocationStrategy();
  }

  auto dynamic_pool = dynamic_cast<umpire::strategy::DynamicPool*>(strategy);

  ASSERT_NE(dynamic_pool, nullptr);

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 0);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  void* alloc1 = nullptr;
  ASSERT_NO_THROW({ alloc1 = alloc.allocate(16); });
  ASSERT_EQ(alloc.getActualSize(), initial_min_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);
  ASSERT_NO_THROW({ alloc.deallocate(alloc1); });
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  ASSERT_NO_THROW({ alloc1 = alloc.allocate(16); });
  ASSERT_EQ(alloc.getActualSize(), initial_min_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  void* alloc2 = nullptr;
  ASSERT_NO_THROW({ alloc2 = alloc.allocate(initial_min_size); });
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);
  ASSERT_EQ(alloc.getActualSize(), 2*initial_min_size);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  void* alloc3 = nullptr;
  ASSERT_NO_THROW({ alloc3 = alloc.allocate(initial_min_size); });
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_min_size);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  ASSERT_NO_THROW({ alloc.deallocate(alloc2); });
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_min_size);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  ASSERT_NO_THROW({ alloc.deallocate(alloc3); });
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_min_size);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 2*initial_min_size);

  ASSERT_NO_THROW({ alloc.deallocate(alloc1); });
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);
}

TEST(HeuristicTest, EdgeCases_0)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto h_fun = umpire::strategy::heuristic_percent_releasable(0);

  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "host_dyn_pool_h_0", rm.getAllocator("HOST"),
      initial_min_size, subsequent_min_size, h_fun);

  auto strategy = alloc.getAllocationStrategy();
  auto tracker = dynamic_cast<umpire::strategy::AllocationTracker*>(strategy);

  if (tracker) {
    strategy = tracker->getAllocationStrategy();
  }

  auto dynamic_pool = dynamic_cast<umpire::strategy::DynamicPool*>(strategy);

  ASSERT_NE(dynamic_pool, nullptr);

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 0);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  void* alloc1 = nullptr;
  ASSERT_NO_THROW({ alloc1 = alloc.allocate(16); });
  ASSERT_EQ(alloc.getActualSize(), initial_min_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  void* alloc2 = nullptr;
  ASSERT_NO_THROW({ alloc2 = alloc.allocate(initial_min_size); });
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);
  ASSERT_EQ(alloc.getActualSize(), 2*initial_min_size);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  void* alloc3 = nullptr;
  ASSERT_NO_THROW({ alloc3 = alloc.allocate(initial_min_size); });
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_min_size);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  ASSERT_NO_THROW({ alloc.deallocate(alloc3); });
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_min_size);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  ASSERT_NO_THROW({ alloc.deallocate(alloc2); });
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_min_size);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 2*initial_min_size);

  ASSERT_NO_THROW({ alloc.deallocate(alloc1); });
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_min_size);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 3*initial_min_size);
}

#if defined(UMPIRE_ENABLE_NUMA)
TEST(NumaPolicyTest, EdgeCases) {
  auto& rm = umpire::ResourceManager::getInstance();

  EXPECT_THROW(rm.makeAllocator<umpire::strategy::NumaPolicy>(
                 "numa_alloc", -1, rm.getAllocator("HOST")),
               umpire::util::Exception);

#if defined(UMPIRE_ENABLE_CUDA)
  const int numa_node = umpire::numa::preferred_node();

  // Only works with HOST allocators
  EXPECT_THROW(rm.makeAllocator<umpire::strategy::NumaPolicy>(
                 "numa_alloc", numa_node, rm.getAllocator("DEVICE")),
               umpire::util::Exception);
#endif
}

TEST(NumaPolicyTest, Location) {
  auto& rm = umpire::ResourceManager::getInstance();

  // TODO Switch this to numa::get_allocatable_nodes() when the issue is fixed
  auto nodes = umpire::numa::get_host_nodes();
  for (auto n : nodes) {
    std::stringstream ss;
    ss << "numa_alloc_" << n;

    auto alloc = rm.makeAllocator<umpire::strategy::NumaPolicy>(
      ss.str(), n, rm.getAllocator("HOST"));

    void* ptr = alloc.allocate(10 * umpire::get_page_size());

    rm.memset(ptr, 0);
    ASSERT_EQ(umpire::numa::get_location(ptr), n);

    alloc.deallocate(ptr);
  }
}

#endif // defined(UMPIRE_ENABLE_NUMA)
