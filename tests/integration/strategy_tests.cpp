//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include <string>
#include <sstream>

#include "umpire/config.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/AlignedAllocator.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/strategy/DynamicPoolHeuristic.hpp"
#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/DynamicPoolMap.hpp"
#include "umpire/strategy/FixedPool.hpp"
#include "umpire/strategy/MixedPool.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/strategy/NamedAllocationStrategy.hpp"
#include "umpire/strategy/QuickPool.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/strategy/SlotPool.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"

#if defined(UMPIRE_ENABLE_NUMA)
#include "umpire/strategy/NumaPolicy.hpp"
#include "umpire/util/numa.hpp"
#endif

#include "umpire/Umpire.hpp"

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <thread>

static int alignment = 16;

static int unique_strategy_id = 0;

static int unique_pool_name = 0;
static int initial_size = 1024;
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
#if defined(UMPIRE_ENABLE_FILE_RESOURCE)
    , "FILE"
#endif
};

template<typename T>
class StrategyTest :
  public ::testing::Test
{
  public:
    void SetUp() override {
      auto& rm = umpire::ResourceManager::getInstance();
      std::string name{"strategy_test_" + std::to_string(unique_strategy_id++)};

      m_allocator = new umpire::Allocator(
          rm.makeAllocator<T>(
            name,
            rm.getAllocator("HOST")));
    }

    void TearDown() override {
      delete m_allocator;
      m_allocator = nullptr;
    }

    umpire::Allocator* m_allocator;

    const std::size_t m_big = 64;
    const std::size_t m_nothing = 0;
};

template<>
void StrategyTest<umpire::strategy::FixedPool>::SetUp()
{
      auto& rm = umpire::ResourceManager::getInstance();
      std::string name{"strategy_test_" + std::to_string(unique_strategy_id++)};

      m_allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::FixedPool>(
            name,
            rm.getAllocator("HOST"),
            m_big*sizeof(double),
            64));
}

#if defined(UMPIRE_ENABLE_CUDA)
template<>
void StrategyTest<umpire::strategy::AllocationAdvisor>::SetUp()
{
      auto& rm = umpire::ResourceManager::getInstance();
      std::string name{"strategy_test_" + std::to_string(unique_strategy_id++)};

      m_allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
            name,
            rm.getAllocator("UM"),
            "READ_MOSTLY"));
}
#endif

template<>
void StrategyTest<umpire::strategy::SizeLimiter>::SetUp()
{
      auto& rm = umpire::ResourceManager::getInstance();
      std::string name{"strategy_test_" + std::to_string(unique_strategy_id++)};

      m_allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::SizeLimiter>(
            name,
            rm.getAllocator("HOST"),
            4*1024));
}

template<>
void StrategyTest<umpire::strategy::SlotPool>::SetUp()
{
      auto& rm = umpire::ResourceManager::getInstance();
      std::string name{"strategy_test_" + std::to_string(unique_strategy_id++)};

      m_allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::SlotPool>(
            name,
            rm.getAllocator("HOST"),
            sizeof(double)));
}

template<>
void StrategyTest<umpire::strategy::MonotonicAllocationStrategy>::SetUp()
{
      auto& rm = umpire::ResourceManager::getInstance();
      std::string name{"strategy_test_" + std::to_string(unique_strategy_id++)};

      m_allocator = new umpire::Allocator(
          rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
            name,
            rm.getAllocator("HOST"),
            4*1024));
}

using Strategies = ::testing::Types<
  umpire::strategy::AlignedAllocator,
#if defined(UMPIRE_ENABLE_CUDA)
  umpire::strategy::AllocationAdvisor,
#endif
  umpire::strategy::DynamicPool,
  umpire::strategy::DynamicPoolList,
  umpire::strategy::DynamicPoolMap,
  umpire::strategy::FixedPool,
  umpire::strategy::MixedPool,
  umpire::strategy::MonotonicAllocationStrategy,
  umpire::strategy::NamedAllocationStrategy,
  umpire::strategy::QuickPool,
  umpire::strategy::SizeLimiter,
  umpire::strategy::SlotPool,
  umpire::strategy::ThreadSafeAllocator>;

TYPED_TEST_SUITE(StrategyTest, Strategies,);

TYPED_TEST(StrategyTest, AllocateDeallocateBig)
{
  double* data = static_cast<double*>(
    this->m_allocator->allocate(this->m_big*sizeof(double)));

  ASSERT_NE(nullptr, data);

  this->m_allocator->deallocate(data);
}

TYPED_TEST(StrategyTest, MultipleAllocateDeallocate)
{
  const int number_of_allocations{8};
  std::vector<void*> allocations;

  for (int i{0}; i < number_of_allocations; ++i) {
    void* ptr = this->m_allocator->allocate(this->m_big*sizeof(double));
    ASSERT_NE(nullptr, ptr);
    allocations.push_back(ptr);
  }

  for (auto ptr : allocations) {
    this->m_allocator->deallocate(ptr);
  }
}

TYPED_TEST(StrategyTest, AllocateDeallocateNothing)
{
  double* data = static_cast<double*>(
    this->m_allocator->allocate(this->m_nothing*sizeof(double)));

  ASSERT_NE(nullptr, data);

  this->m_allocator->deallocate(data);
}

TYPED_TEST(StrategyTest, DeallocateNullptr)
{
  double* data = nullptr;

  ASSERT_NO_THROW(this->m_allocator->deallocate(data));

  SUCCEED();
}

TYPED_TEST(StrategyTest, GetSize)
{
  const std::size_t size = this->m_big*sizeof(double);

  double* data = static_cast<double*>(
    this->m_allocator->allocate(size));

  ASSERT_EQ(size, this->m_allocator->getSize(data));

  this->m_allocator->deallocate(data);
}

TYPED_TEST(StrategyTest, GetById)
{
  auto& rm = umpire::ResourceManager::getInstance();

  int id = this->m_allocator->getId();
  ASSERT_GE(id, 0);

  auto allocator_by_id = rm.getAllocator(id);

  ASSERT_EQ(this->m_allocator->getAllocationStrategy(), allocator_by_id.getAllocationStrategy());

  ASSERT_THROW(
      rm.getAllocator(-25),
      umpire::util::Exception);
}

TYPED_TEST(StrategyTest, get_allocator_records)
{
  double* data = static_cast<double*>(
    this->m_allocator->allocate(this->m_big*sizeof(double)));

  auto records = umpire::get_allocator_records(*(this->m_allocator));

  ASSERT_EQ(records.size(), 1);

  this->m_allocator->deallocate(data);
}

TYPED_TEST(StrategyTest, getCurrentSize)
{
  ASSERT_EQ(this->m_allocator->getCurrentSize(), 0);

  void* data = this->m_allocator->allocate(this->m_big*sizeof(double));

  ASSERT_EQ(this->m_allocator->getCurrentSize(), this->m_big*sizeof(double));

  this->m_allocator->deallocate(data);
}

TYPED_TEST(StrategyTest, getActualSize)
{
  void* data = this->m_allocator->allocate(this->m_big*sizeof(double));
  ASSERT_GE(this->m_allocator->getActualSize(), this->m_big*sizeof(double));

  this->m_allocator->deallocate(data);
}

class DynamicPoolTest :
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
           , initial_size
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

TEST_P(DynamicPoolTest, Allocate) {
  void* alloc = nullptr;
  alloc = allocator->allocate(100);
  allocator->deallocate(alloc);
}

TEST_P(DynamicPoolTest, Sizes) {
  void* alloc = nullptr;
  ASSERT_NO_THROW({ alloc = allocator->allocate(100); });
  ASSERT_EQ(allocator->getSize(alloc), 100);
  ASSERT_GE(allocator->getCurrentSize(), 100);
  ASSERT_EQ(allocator->getHighWatermark(), 100);
  ASSERT_GE(allocator->getActualSize(), initial_size);

  void* alloc2 = nullptr;
  ASSERT_NO_THROW({ alloc2 = allocator->allocate(initial_size); });
  ASSERT_NO_THROW({ allocator->deallocate(alloc); });

  ASSERT_GE(allocator->getCurrentSize(), initial_size);
  ASSERT_EQ(allocator->getHighWatermark(), initial_size+100);
  ASSERT_GE(allocator->getActualSize(), initial_size+subsequent_min_size);
  ASSERT_EQ(allocator->getSize(alloc2), initial_size);

  ASSERT_NO_THROW({ allocator->deallocate(alloc2); });
}

TEST_P(DynamicPoolTest, Duplicate)
{
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_TRUE(rm.isAllocator(allocatorName));

  ASSERT_EQ(allocator->getName(), poolName.str());

  ASSERT_ANY_THROW(
      rm.makeAllocator<umpire::strategy::DynamicPool>(
        poolName.str(), rm.getAllocator(allocatorName)));
}

INSTANTIATE_TEST_SUITE_P(Allocations, DynamicPoolTest, ::testing::ValuesIn(AllocationDevices));

class DynamicPoolListTest :
  public ::testing::TestWithParam<const char*>
{
  public:
    void SetUp() override {
      auto& rm = umpire::ResourceManager::getInstance();
      allocatorName = GetParam();
      poolName << allocatorName << "_pool_" << unique_pool_name++;

      rm.makeAllocator<umpire::strategy::DynamicPoolList>
        (  poolName.str()
           , rm.getAllocator(allocatorName)
           , initial_size
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

TEST_P(DynamicPoolListTest, Allocate) {
  void* alloc = nullptr;
  alloc = allocator->allocate(100);
  allocator->deallocate(alloc);
}

TEST_P(DynamicPoolListTest, Sizes) {
  void* alloc = nullptr;
  ASSERT_NO_THROW({ alloc = allocator->allocate(100); });
  ASSERT_EQ(allocator->getSize(alloc), 100);
  ASSERT_GE(allocator->getCurrentSize(), 100);
  ASSERT_EQ(allocator->getHighWatermark(), 100);
  ASSERT_GE(allocator->getActualSize(), initial_size);

  void* alloc2 = nullptr;
  ASSERT_NO_THROW({ alloc2 = allocator->allocate(initial_size); });
  ASSERT_NO_THROW({ allocator->deallocate(alloc); });

  ASSERT_GE(allocator->getCurrentSize(), initial_size);
  ASSERT_EQ(allocator->getHighWatermark(), initial_size+100);
  ASSERT_GE(allocator->getActualSize(), initial_size+subsequent_min_size);
  ASSERT_EQ(allocator->getSize(alloc2), initial_size);

  ASSERT_NO_THROW({ allocator->deallocate(alloc2); });
}

TEST_P(DynamicPoolListTest, Duplicate)
{
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_TRUE(rm.isAllocator(allocatorName));

  ASSERT_EQ(allocator->getName(), poolName.str());

  ASSERT_ANY_THROW(
      rm.makeAllocator<umpire::strategy::DynamicPoolList>(
        poolName.str(), rm.getAllocator(allocatorName)));
}

INSTANTIATE_TEST_SUITE_P(Allocations, DynamicPoolListTest, ::testing::ValuesIn(AllocationDevices));

TEST(DynamicPool, LimitedResource)
{
  auto& rm = umpire::ResourceManager::getInstance();

  const std::size_t max_mem{1024 * 1024 * 4};

  auto limited_resource = rm.makeAllocator<umpire::strategy::SizeLimiter>(
    "limited_resource", rm.getAllocator("HOST"), max_mem);

  auto allocator = rm.makeAllocator<umpire::strategy::DynamicPool>(
    "host_dyn_pool", limited_resource, 0, 1024);

  ASSERT_EQ(allocator.getName(), "host_dyn_pool");

  void* alloc{nullptr};

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
    alloc_size += 1024;
  }

  ASSERT_NO_THROW( { allocator.deallocate(alloc3); } );
}

TEST(MonotonicStrategy, Host)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
      "host_monotonic_pool", rm.getAllocator("HOST"), 65536);

  void* alloc = allocator.allocate(100);
  void* alloc2 = allocator.allocate(100);

  ASSERT_EQ(static_cast<char*>(alloc2) - static_cast<char*>(alloc), 100);
  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_EQ(allocator.getName(), "host_monotonic_pool");

  allocator.deallocate(alloc2);
  allocator.deallocate(alloc);
}

#if defined(UMPIRE_ENABLE_DEVICE)
TEST(MonotonicStrategy, Device)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
      "device_monotonic_pool", rm.getAllocator("DEVICE"), 65536);

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
      "um_monotonic_pool", rm.getAllocator("UM"), 65536);

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

  const std::size_t max_power = 9;
  void* alloc[max_power];
  std::size_t size = 4, total_size = 0;
  for (std::size_t i = 0; i < max_power; ++i) {
    alloc[i] = allocator.allocate(size);
    total_size += size;
    size *= 4;
  }

  ASSERT_EQ(allocator.getCurrentSize(), total_size);
  ASSERT_GT(allocator.getActualSize(), total_size);
  ASSERT_EQ(allocator.getSize(alloc[max_power-1]), size/4);
  ASSERT_GE(allocator.getHighWatermark(), total_size);
  ASSERT_EQ(allocator.getName(), "host_mixed_pool");

  for (std::size_t i = 0; i < max_power; ++i)
    allocator.deallocate(alloc[i]);
}

TEST(ThreadSafeAllocator, HostStdThread)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      "thread_safe_allocator_host_std", rm.getAllocator("HOST"));

  constexpr int N = 16;
  std::vector<void*> thread_allocs{N};
  std::vector<std::thread> threads;

  for (std::size_t i = 0; i < N; ++i)
  {
    threads.push_back(
        std::thread([=, &allocator, &thread_allocs] {
          for ( int j = 0; j < N; ++j) {
            thread_allocs[i] = allocator.allocate(1024);
            ASSERT_NE(thread_allocs[i], nullptr);
            allocator.deallocate(thread_allocs[i]);
            thread_allocs[i] = allocator.allocate(1024);
            ASSERT_NE(thread_allocs[i], nullptr);
          }
    }));
  }

  for (auto& t : threads) {
    t.join();
  }

  for (auto alloc : thread_allocs)
  {
    ASSERT_NE(alloc, nullptr);
  }

  ASSERT_NO_THROW({
    for (auto alloc : thread_allocs)
    {
      allocator.deallocate(alloc);
    }
  });
}

#if defined(_OPENMP)
TEST(ThreadSafeAllocator, HostOpenMP)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      "thread_safe_allocator_host_omp", rm.getAllocator("HOST"));

#pragma omp parallel
  {
    const std::size_t size = 1024*omp_get_thread_num();

    double* thread_data = static_cast<double*>(
     allocator.allocate(size*sizeof(double)));

    allocator.deallocate(thread_data);
  }

  SUCCEED();
}
#endif

#if defined(UMPIRE_ENABLE_DEVICE)
TEST(ThreadSafeAllocator, DeviceStdThread)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      "thread_safe_allocator_device_std", rm.getAllocator("DEVICE"));

  constexpr int N = 16;
  std::vector<void*> thread_allocs{N};
  std::vector<std::thread> threads;

  for (std::size_t i = 0; i < N; ++i)
  {
    threads.push_back(
        std::thread([=, &allocator, &thread_allocs] {
          for ( int j = 0; j < N; ++j) {
            thread_allocs[i] = allocator.allocate(1024);
            ASSERT_NE(thread_allocs[i], nullptr);
            allocator.deallocate(thread_allocs[i]);
            thread_allocs[i] = allocator.allocate(1024);
            ASSERT_NE(thread_allocs[i], nullptr);
          }
    }));
  }

  for (auto& t : threads) {
    t.join();
  }

  for (auto alloc : thread_allocs)
  {
    ASSERT_NE(alloc, nullptr);
  }

  ASSERT_NO_THROW({
    for (auto alloc : thread_allocs)
    {
      allocator.deallocate(alloc);
    }
  });
}

#if defined(_OPENMP)
TEST(ThreadSafeAllocator, DeviceOpenMP)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      "thread_safe_allocator_device_omp", rm.getAllocator("DEVICE"));

#pragma omp parallel
  {
    const std::size_t size = 1024*omp_get_thread_num();

    double* thread_data = static_cast<double*>(
     allocator.allocate(size*sizeof(double)));

    allocator.deallocate(thread_data);
  }

  SUCCEED();
}
#endif
#endif // defined(UMPIRE_ENABLE_DEVICE)


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

TEST(DynamicPoolMapReleaseTest, Works)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPoolMap>(
      "host_dyn_pool_map_for_release", rm.getAllocator("HOST"), 64, 64);

  void* ptr_one = alloc.allocate(62);
  void* ptr_two = alloc.allocate(1024);
  alloc.deallocate(ptr_two);

  ASSERT_EQ(alloc.getCurrentSize(), 62);
  EXPECT_NO_THROW(alloc.release());

  ASSERT_LE(alloc.getActualSize(), 64);

  alloc.deallocate(ptr_one);
  ASSERT_EQ(alloc.getCurrentSize(), 0);

  EXPECT_NO_THROW(alloc.release());

  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_LE(alloc.getActualSize(), 0);
}

TEST(DynamicPoolMapReleaseTest, MissingBlocks)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::DynamicPoolMap>(
      "host_dyn_pool_map_for_release_2", rm.getAllocator("HOST"), 128, 64);

  void* data_one = allocator.allocate(128);
  void* data_two = allocator.allocate(44);

  allocator.deallocate(data_one);
  allocator.deallocate(data_two);

  ASSERT_EQ(allocator.getCurrentSize(), 0);
  ASSERT_GE(allocator.getActualSize(), 0);

  allocator.release();

  ASSERT_EQ(allocator.getCurrentSize(), 0);
  ASSERT_EQ(allocator.getActualSize(), 0);
}

TEST(DynamicPoolListReleaseTest, Works)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPoolList>(
      "host_dyn_pool_list_for_release", rm.getAllocator("HOST"), 64, 64);

  void* ptr_one = alloc.allocate(62);
  void* ptr_two = alloc.allocate(1024);
  alloc.deallocate(ptr_two);

  ASSERT_EQ(alloc.getCurrentSize(), 62);
  EXPECT_NO_THROW(alloc.release());

  ASSERT_LE(alloc.getActualSize(), 64);

  alloc.deallocate(ptr_one);
  ASSERT_EQ(alloc.getCurrentSize(), 0);

  EXPECT_NO_THROW(alloc.release());

  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_LE(alloc.getActualSize(), 0);
}

TEST(DynamicPoolListReleaseTest, MissingBlocks)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::DynamicPoolList>(
      "host_dyn_pool_list_for_release_2", rm.getAllocator("HOST"), 128, 64);

  void* data_one = allocator.allocate(128);
  void* data_two = allocator.allocate(44);

  allocator.deallocate(data_one);
  allocator.deallocate(data_two);

  ASSERT_EQ(allocator.getCurrentSize(), 0);
  ASSERT_GE(allocator.getActualSize(), 0);

  allocator.release();

  ASSERT_EQ(allocator.getCurrentSize(), 0);
  ASSERT_EQ(allocator.getActualSize(), 0);
}

TEST(DynamicPoolList, largestavailable)
{
  auto& rm = umpire::ResourceManager::getInstance();
  const int num_allocs = 1024;

  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPoolList>(
    "host_dyn_pool_list_for_largestavailable", rm.getAllocator("HOST"),
    num_allocs * 1024);

  auto dynamic_pool =
    umpire::util::unwrap_allocator<umpire::strategy::DynamicPoolList>(alloc);

  ASSERT_NE(dynamic_pool, nullptr);

  alloc.deallocate(alloc.allocate(1024));

  ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(), num_allocs * 1024);

  void* ptrs[num_allocs];

  for ( int i{0}; i < num_allocs; ++i ) {
    ptrs[i] = alloc.allocate(1024);
    ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(), ( (num_allocs-(i+1)) * 1024) );
  }

  for ( int i{0}; i < num_allocs; i += 2 ) {
    alloc.deallocate(ptrs[i]);
    ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(), 1024);
  }

  for ( int i{1}; i < num_allocs; i += 2 ) {
    const int largest_block = ((i+2) < num_allocs) ? (i+2) * 1024 : (i+1) * 1024;
    alloc.deallocate(ptrs[i]);
    ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(), largest_block);
  }
}

TEST(DynamicPoolMap, largestavailable)
{
  auto& rm = umpire::ResourceManager::getInstance();
  const int num_allocs = 1024;

  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPoolMap>(
    "host_dyn_pool_map_for_largestavailable", rm.getAllocator("HOST"),
    num_allocs * 1024);

  auto dynamic_pool =
    umpire::util::unwrap_allocator<umpire::strategy::DynamicPoolMap>(alloc);

  ASSERT_NE(dynamic_pool, nullptr);

  alloc.deallocate(alloc.allocate(1024));

  ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(), num_allocs * 1024);

  void* ptrs[num_allocs];

  for ( int i{0}; i < num_allocs; ++i ) {
    ptrs[i] = alloc.allocate(1024);
    ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(), ( (num_allocs-(i+1)) * 1024) );
  }

  for ( int i{0}; i < num_allocs; i += 2 ) {
    alloc.deallocate(ptrs[i]);
    ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(), 1024);
  }

  for ( int i{1}; i < num_allocs; i += 2 ) {
    const int largest_block = ((i+2) < num_allocs) ? (i+2) * 1024 : (i+1) * 1024;
    alloc.deallocate(ptrs[i]);
    ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(), largest_block);
  }
}

TEST(DynamicPoolMap, coalesce)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPoolMap>(
    "host_dyn_pool_map_for_coalesce", rm.getAllocator("HOST"));

  auto dynamic_pool =
    umpire::util::unwrap_allocator<umpire::strategy::DynamicPoolMap>(alloc);

  ASSERT_NE(dynamic_pool, nullptr);

  const std::size_t initial_bytes{dynamic_pool->getActualSize()};

  void* ptr_one = alloc.allocate(62);
  void* ptr_two = alloc.allocate(1024);

  ASSERT_GT(dynamic_pool->getBlocksInPool(), 2);
  alloc.deallocate(ptr_two);

  dynamic_pool->coalesce();

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);

  alloc.deallocate(ptr_one);

  dynamic_pool->coalesce();

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);

  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(dynamic_pool->getActualSize(), initial_bytes);
  ASSERT_GE(alloc.getHighWatermark(), 62 + 1024);
}

TEST(DynamicPoolList, coalesce)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPoolList>(
    "host_dyn_pool_list_for_coalesce", rm.getAllocator("HOST"), 128, 1024);

  auto dynamic_pool =
    umpire::util::unwrap_allocator<umpire::strategy::DynamicPoolList>(alloc);

  ASSERT_NE(dynamic_pool, nullptr);

  void* ptr_one = alloc.allocate(62);

  const std::size_t initial_bytes{dynamic_pool->getActualSize()};

  void* ptr_two = alloc.allocate(1024);

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);
  alloc.deallocate(ptr_two);

  dynamic_pool->coalesce();

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);

  alloc.deallocate(ptr_one);

  dynamic_pool->coalesce();

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);

  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(dynamic_pool->getActualSize(), initial_bytes+1024);
  ASSERT_GE(alloc.getHighWatermark(), 62 + 1024);
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
      1024ul, 1024ul, alignment, h_fun);

  auto dynamic_pool = umpire::util::unwrap_allocator<umpire::strategy::DynamicPool>(alloc);

  ASSERT_NE(dynamic_pool, nullptr);

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 1024);

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
      initial_size, subsequent_min_size, alignment, h_fun);

  auto dynamic_pool = umpire::util::unwrap_allocator<umpire::strategy::DynamicPool>(alloc);

  ASSERT_NE(dynamic_pool, nullptr);

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), initial_size);

  void* alloc1{nullptr};
  ASSERT_NO_THROW({ alloc1 = alloc.allocate(16); });
  ASSERT_EQ(alloc.getActualSize(), initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);
  ASSERT_NO_THROW({ alloc.deallocate(alloc1); });
  ASSERT_NO_THROW(dynamic_pool->coalesce());
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), initial_size);

  ASSERT_NO_THROW({ alloc1 = alloc.allocate(16); });
  ASSERT_EQ(alloc.getCurrentSize(), 16);
  ASSERT_EQ(alloc.getActualSize(), initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  void* alloc2{nullptr};
  ASSERT_NO_THROW({ alloc2 = alloc.allocate(initial_size); });
  ASSERT_EQ(alloc.getCurrentSize(), initial_size+16);
  ASSERT_EQ(alloc.getActualSize(), 2*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  void* alloc3{nullptr};
  ASSERT_NO_THROW({ alloc3 = alloc.allocate(initial_size); });
  ASSERT_EQ(alloc.getCurrentSize(), 2*initial_size+16);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 4);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  ASSERT_NO_THROW({ alloc.deallocate(alloc2); });
  ASSERT_EQ(alloc.getCurrentSize(), initial_size+16);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 4);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), initial_size);

  ASSERT_NO_THROW({ alloc.deallocate(alloc3); });
  ASSERT_EQ(alloc.getCurrentSize(), 16);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 4);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 2*initial_size);

  ASSERT_NO_THROW({ alloc.deallocate(alloc1); });
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_size);

  ASSERT_NO_THROW({ dynamic_pool->coalesce(); });
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_size);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 3*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
}

TEST(HeuristicTest, EdgeCases_0)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto h_fun = umpire::strategy::heuristic_percent_releasable(0);

  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPool>(
      "host_dyn_pool_h_0", rm.getAllocator("HOST"),
      initial_size, subsequent_min_size, alignment, h_fun);

  auto dynamic_pool = umpire::util::unwrap_allocator<umpire::strategy::DynamicPool>(alloc);

  ASSERT_NE(dynamic_pool, nullptr);

  //
  // After construction, we expect the pool to look like:
  // getCurrentSize     == 0
  // getActualSize      == 1*initial_size
  // getBlocksInPool    == 1:
  //    block #1 (Whole Block: free(initial_size))
  // getReleaseableSize == 1*initial_size
  //
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(alloc.getActualSize(), initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), initial_size);

  //
  // After alloc1=allocate(16), we expect the pool to look like:
  // getCurrentSize     == 16
  // getActualSize      == 1*initial_size
  // getBlocksInPool    == 2:
  //    block #1 (Partial Block: alloc1(16))
  //    block #2 (Partial Block: free(initial_size-16))
  // getReleaseableSize == 0
  //
  void* alloc1 = nullptr;
  ASSERT_NO_THROW({ alloc1 = alloc.allocate(16); });
  ASSERT_EQ(alloc.getCurrentSize(), 16);
  ASSERT_EQ(alloc.getActualSize(), initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  //
  // After alloc2=allocate(initial_size), we expect the pool to look like:
  // getCurrentSize     == 16+(1*initial_size)
  // getActualSize      == 2*initial_size
  // getBlocksInPool    == 3:
  //    block #1 (Partial Block: alloc1(16))
  //    block #2 (Partial Block: free(initial_size-16))
  //    block #3 (Whole Block: alloc2(initial_size))
  // getReleaseableSize == 0
  //
  void* alloc2 = nullptr;
  ASSERT_NO_THROW({ alloc2 = alloc.allocate(initial_size); });
  ASSERT_EQ(alloc.getCurrentSize(), 16+initial_size);
  ASSERT_EQ(alloc.getActualSize(), 2*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  //
  // After alloc3=allocate(initial_size), we expect the pool to look like:
  // getCurrentSize     == 16+(2*initial_size)
  // getActualSize      == 3*initial_size
  // getBlocksInPool    == 4:
  //    block #1 (Partial Block: alloc1(16))
  //    block #2 (Partial Block: free(initial_size-16))
  //    block #3 (Whole Block: alloc2(initial_size))
  //    block #4 (Whole Block: alloc3(initial_size))
  // getReleaseableSize == 0
  //
  void* alloc3 = nullptr;
  ASSERT_NO_THROW({ alloc3 = alloc.allocate(initial_size); });
  ASSERT_EQ(alloc.getCurrentSize(), 16+(2*initial_size));
  ASSERT_EQ(alloc.getActualSize(), 3*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 4);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  //
  // After deallocate(alloc3), we expect the pool to look like:
  // getCurrentSize     == 16+(1*initial_size)
  // getActualSize      == 3*initial_size
  // getBlocksInPool    == 3:
  //    block #1 (Partial Block: alloc1(16))
  //    block #2 (Partial Block: free(initial_size-16))
  //    block #3 (Whole Block: alloc2(initial_size))
  //    block #4 (Whole Block: free(initial_size))
  // getReleaseableSize == 1*initial_size
  //
  ASSERT_NO_THROW({ alloc.deallocate(alloc3); });
  ASSERT_EQ(alloc.getCurrentSize(), 16+(1*initial_size));
  ASSERT_EQ(alloc.getActualSize(), 3*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 4);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), initial_size);

  //
  // After deallocate(alloc2), we expect the pool to look like:
  // getCurrentSize     == 16
  // getActualSize      == 3*initial_size
  // getBlocksInPool    == 3:
  //    block #1 (Partial Block: alloc1(16))
  //    block #2 (Partial Block: free(initial_size-16))
  //    block #3 (Whole Block: free(initial_size))
  //    block #4 (Whole Block: free(initial_size))
  // getReleaseableSize == 2*initial_size
  //
  ASSERT_NO_THROW({ alloc.deallocate(alloc2); });
  ASSERT_EQ(alloc.getCurrentSize(), 16);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 4);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 2*initial_size);

  //
  // After deallocate(alloc1), we expect the pool to look like:
  // getCurrentSize     == 0
  // getActualSize      == 3*initial_size
  // getBlocksInPool    == 4:
  //    block #1 (Partial Block: free(16))
  //    block #2 (Partial Block: free(initial_size-16))
  //    block #3 (Whole Block: free(initial_size))
  //    block #4 (Whole Block: free(initial_size))
  // getReleaseableSize == 2*initial_size
  //
  ASSERT_NO_THROW({ alloc.deallocate(alloc1); });
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 4);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 2*initial_size);

  //
  // After release, we expect the pool to look like:
  // getCurrentSize     == 0
  // getActualSize      == 0
  // getBlocksInPool    == 0
  // getReleaseableSize == 0
  //
  EXPECT_NO_THROW(alloc.release());
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(alloc.getActualSize(), 0);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 0);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  //
  // final_alloc final_alloc=allocate(16), we expect the pool to look like:
  // getCurrentSize     == 16
  // getActualSize      == initial_size
  // getBlocksInPool    == 2:
  //    block #1 (Partial Block: final_alloc(16))
  //    block #2 (Partial Block: free(initial_size-16))
  // getReleaseableSize == 0
  //
  void* final_alloc = nullptr;
  ASSERT_NO_THROW({ final_alloc = alloc.allocate(16); });
  ASSERT_EQ(alloc.getCurrentSize(), 16);
  ASSERT_EQ(alloc.getActualSize(), initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  //
  // After deallocate(final_alloc), we expect the pool to look like:
  // getCurrentSize     == 0
  // getActualSize      == initial_size
  // getBlocksInPool    == 2:
  //    block #1 (Partial Block: free(16))
  //    block #2 (Partial Block: free(initial_size))
  // getReleaseableSize == 0*initial_size
  //
  ASSERT_NO_THROW({ alloc.deallocate(final_alloc); });
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(alloc.getActualSize(), initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);
}

TEST(HeuristicTestList, EdgeCases_0)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto h_fun = umpire::strategy::heuristic_percent_releasable_list(0);

  auto alloc = rm.makeAllocator<umpire::strategy::DynamicPoolList>(
      "host_dyn_pool__list_h_0", rm.getAllocator("HOST"),
      initial_size, subsequent_min_size, h_fun);

  auto dynamic_pool = umpire::util::unwrap_allocator<umpire::strategy::DynamicPoolList>(alloc);

  ASSERT_NE(dynamic_pool, nullptr);

  //
  // After construction, we expect the pool to look like:
  // getCurrentSize     == 0
  // getActualSize      == 0
  // getBlocksInPool    == 0
  // getReleaseableSize == 0
  //
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(alloc.getActualSize(), 0);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 0);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  //
  // After alloc1=allocate(16), we expect the pool to look like:
  // getCurrentSize     == 16
  // getActualSize      == 1*initial_size
  // getBlocksInPool    == 1: (whole blocks, not counting fragments)
  // getReleaseableSize == 0
  //
  void* alloc1 = nullptr;
  ASSERT_NO_THROW({ alloc1 = alloc.allocate(16); });
  ASSERT_EQ(alloc.getCurrentSize(), 16);
  ASSERT_EQ(alloc.getActualSize(), initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  //
  // After alloc2=allocate(initial_size), we expect the pool to look like:
  // getCurrentSize     == 16+(1*initial_size)
  // getActualSize      == 2*initial_size
  // getBlocksInPool    == 2:
  //    block #1 (Partial Block: alloc1(16))
  //    block #1 (Partial Block: free(initial_size-16))
  //    block #2 (Whole Block: alloc2(initial_size))
  // getReleaseableSize == 0
  //
  void* alloc2 = nullptr;
  ASSERT_NO_THROW({ alloc2 = alloc.allocate(initial_size); });
  ASSERT_EQ(alloc.getCurrentSize(), 16+initial_size);
  ASSERT_EQ(alloc.getActualSize(), 2*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  //
  // After alloc3=allocate(initial_size), we expect the pool to look like:
  // getCurrentSize     == 16+(2*initial_size)
  // getActualSize      == 3*initial_size
  // getBlocksInPool    == 3:
  //    block #1 (Partial Block: alloc1(16))
  //    block #1 (Partial Block: free(initial_size-16))
  //    block #2 (Whole Block: alloc2(initial_size))
  //    block #3 (Whole Block: alloc3(initial_size))
  // getReleaseableSize == 0
  //
  void* alloc3 = nullptr;
  ASSERT_NO_THROW({ alloc3 = alloc.allocate(initial_size); });
  ASSERT_EQ(alloc.getCurrentSize(), 16+(2*initial_size));
  ASSERT_EQ(alloc.getActualSize(), 3*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  //
  // After deallocate(alloc3), we expect the pool to look like:
  // getCurrentSize     == 16+(1*initial_size)
  // getActualSize      == 3*initial_size
  // getBlocksInPool    == 3:
  //    block #1 (Partial Block: alloc1(16))
  //    block #1 (Partial Block: free(initial_size-16))
  //    block #2 (Whole Block: alloc2(initial_size))
  //    block #3 (Whole Block: free(initial_size))
  // getReleaseableSize == 1*initial_size
  //
  ASSERT_NO_THROW({ alloc.deallocate(alloc3); });
  ASSERT_EQ(alloc.getCurrentSize(), 16+(1*initial_size));
  ASSERT_EQ(alloc.getActualSize(), 3*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  //
  // After deallocate(alloc2), we expect the pool to look like:
  // getCurrentSize     == 16
  // getActualSize      == 3*initial_size
  // getBlocksInPool    == 3:
  //    block #1 (Partial Block: alloc1(16))
  //    block #1 (Partial Block: free(initial_size-16))
  //    block #2 (Whole Block: free(initial_size))
  //    block #3 (Whole Block: free(initial_size))
  // getReleaseableSize == 2*initial_size
  //
  ASSERT_NO_THROW({ alloc.deallocate(alloc2); });
  ASSERT_EQ(alloc.getCurrentSize(), 16);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 2*initial_size);

  //
  // After deallocate(alloc1), we expect the pool to look like:
  // getCurrentSize     == 0
  // getActualSize      == 3*initial_size
  // getBlocksInPool    == 3:
  //    block #1 (Whole Block: free(initial_size))
  //    block #2 (Whole Block: free(initial_size))
  //    block #3 (Whole Block: free(initial_size))
  // getReleaseableSize == 2*initial_size
  //
  ASSERT_NO_THROW({ alloc.deallocate(alloc1); });
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(alloc.getActualSize(), 3*initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 3*initial_size);

  //
  // After release, we expect the pool to look like:
  // getCurrentSize     == 0
  // getActualSize      == 0
  // getBlocksInPool    == 0
  // getReleaseableSize == 0
  //
  EXPECT_NO_THROW(alloc.release());
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(alloc.getActualSize(), 0);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 0);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  //
  // final_alloc final_alloc=allocate(16), we expect the pool to look like:
  // getCurrentSize     == 16
  // getActualSize      == initial_size
  // getBlocksInPool    == 1:
  //    block #1 (Partial Block: final_alloc(16))
  //    block #1 (Partial Block: free(initial_size-16))
  // getReleaseableSize == 0
  //
  void* final_alloc = nullptr;
  ASSERT_NO_THROW({ final_alloc = alloc.allocate(16); });
  ASSERT_EQ(alloc.getCurrentSize(), 16);
  ASSERT_EQ(alloc.getActualSize(), initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);

  //
  // After deallocate(final_alloc), we expect the pool to look like:
  // getCurrentSize     == 0
  // getActualSize      == initial_size
  // getBlocksInPool    == 1:
  //    block #1 (Whole Block: free(initial_size))
  // getReleaseableSize == 0*initial_size
  //
  ASSERT_NO_THROW({ alloc.deallocate(final_alloc); });
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(alloc.getActualSize(), initial_size);
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);
  ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);
}

#if defined(UMPIRE_ENABLE_NUMA)
TEST(NumaPolicyTest, EdgeCases) {
  auto& rm = umpire::ResourceManager::getInstance();

  EXPECT_THROW(rm.makeAllocator<umpire::strategy::NumaPolicy>(
                 "numa_alloc", rm.getAllocator("HOST"), -1),
               umpire::util::Exception);

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
  const int numa_node = umpire::numa::preferred_node();

  // Only works with HOST allocators
  EXPECT_THROW(rm.makeAllocator<umpire::strategy::NumaPolicy>(
                 "numa_alloc", rm.getAllocator("DEVICE"), numa_node),
               umpire::util::Exception);
#endif
}

TEST(NumaPolicyTest, Location) {
  auto& rm = umpire::ResourceManager::getInstance();

  auto nodes = umpire::numa::get_host_nodes();
  for (auto n : nodes) {
    std::stringstream ss;
    ss << "numa_alloc_" << n;

    auto alloc = rm.makeAllocator<umpire::strategy::NumaPolicy>(
      ss.str(), rm.getAllocator("HOST"), n);

    void* ptr = alloc.allocate(10 * umpire::get_page_size());

    rm.memset(ptr, 0);
    ASSERT_EQ(umpire::numa::get_location(ptr), n);

    alloc.deallocate(ptr);
  }
}

#endif // defined(UMPIRE_ENABLE_NUMA)

static inline void test_alignment(
    uintptr_t p,
    unsigned int align)
{
  ASSERT_EQ(0, p % align);
  p &= align;
  ASSERT_TRUE( p == 0 || p == align);
}

TEST(AlignedAllocator, AllocateAlign256)
{
  unsigned int align = 256;
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.makeAllocator<umpire::strategy::AlignedAllocator>(
    "aligned_allocator_256", rm.getAllocator("HOST"), align);

  void* d1{alloc.allocate(1)};
  void* d2{alloc.allocate(257)};
  void* d3{alloc.allocate(783)};

  test_alignment(
    reinterpret_cast<uintptr_t>(d1),
    align);
  test_alignment(
    reinterpret_cast<uintptr_t>(d2),
    align);
  test_alignment(
    reinterpret_cast<uintptr_t>(d3),
    align);

  alloc.deallocate(d1);
  alloc.deallocate(d2);
  alloc.deallocate(d3);
}

TEST(AlignedAllocator, AllocateAlign64)
{
  unsigned int align = 64;
  auto& rm = umpire::ResourceManager::getInstance();
  auto alloc = rm.makeAllocator<umpire::strategy::AlignedAllocator>(
    "aligned_allocator_64", rm.getAllocator("HOST"), align);

  void* d1{alloc.allocate(1)};
  void* d2{alloc.allocate(17)};
  void* d3{alloc.allocate(128)};

  test_alignment(
    reinterpret_cast<uintptr_t>(d1),
    align);
  test_alignment(
    reinterpret_cast<uintptr_t>(d2),
    align);
  test_alignment(
    reinterpret_cast<uintptr_t>(d3),
    align);

  alloc.deallocate(d1);
  alloc.deallocate(d2);
  alloc.deallocate(d3);
}

TEST(AlignedAllocator, BadAlignment)
{
  auto& rm = umpire::ResourceManager::getInstance();

  EXPECT_THROW({
    auto alloc = rm.makeAllocator<umpire::strategy::AlignedAllocator>(
    "aligned_allocator_6", rm.getAllocator("HOST"), 6);
    UMPIRE_USE_VAR(alloc);
  }, umpire::util::Exception);

  EXPECT_THROW({
    auto alloc = rm.makeAllocator<umpire::strategy::AlignedAllocator>(
    "aligned_allocator_11", rm.getAllocator("HOST"), 11);
    UMPIRE_USE_VAR(alloc);
  }, umpire::util::Exception);

  EXPECT_THROW({
    auto alloc = rm.makeAllocator<umpire::strategy::AlignedAllocator>(
    "aligned_allocator_0", rm.getAllocator("HOST"), 0);
    UMPIRE_USE_VAR(alloc);
  }, umpire::util::Exception);
}
