//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "umpire/ResourceManager.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/AlignedAllocator.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
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

static int unique_strategy_id = 0;

const char* AllocationDevices[] = {"HOST"
#if defined(UMPIRE_ENABLE_DEVICE)
                                   ,
                                   "DEVICE"
#endif
#if defined(UMPIRE_ENABLE_UM)
                                   ,
                                   "UM"
#endif
#if defined(UMPIRE_ENABLE_PINNED)
                                   ,
                                   "PINNED"
#endif
#if defined(UMPIRE_ENABLE_FILE_RESOURCE)
                                   ,
                                   "FILE"
#endif
};

template <typename T>
class StrategyTest : public ::testing::Test {
 public:
  void SetUp() override
  {
    auto& rm = umpire::ResourceManager::getInstance();
    std::string name{"strategy_test_" + std::to_string(unique_strategy_id++)};

    m_allocator = new umpire::Allocator(
        rm.makeAllocator<T>(name, rm.getAllocator("HOST")));
  }

  void TearDown() override
  {
    delete m_allocator;
    m_allocator = nullptr;
  }

  umpire::Allocator* m_allocator;

  const std::size_t m_big = 64;
  const std::size_t m_nothing = 0;
};

template <>
void StrategyTest<umpire::strategy::FixedPool>::SetUp()
{
  auto& rm = umpire::ResourceManager::getInstance();
  std::string name{"strategy_test_" + std::to_string(unique_strategy_id++)};

  m_allocator =
      new umpire::Allocator(rm.makeAllocator<umpire::strategy::FixedPool>(
          name, rm.getAllocator("HOST"), m_big * sizeof(double), 64));
}

#if defined(UMPIRE_ENABLE_CUDA)
template <>
void StrategyTest<umpire::strategy::AllocationAdvisor>::SetUp()
{
  auto& rm = umpire::ResourceManager::getInstance();
  std::string name{"strategy_test_" + std::to_string(unique_strategy_id++)};

  m_allocator = new umpire::Allocator(
      rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
          name, rm.getAllocator("UM"), "READ_MOSTLY"));
}
#endif

template <>
void StrategyTest<umpire::strategy::SizeLimiter>::SetUp()
{
  auto& rm = umpire::ResourceManager::getInstance();
  std::string name{"strategy_test_" + std::to_string(unique_strategy_id++)};

  m_allocator =
      new umpire::Allocator(rm.makeAllocator<umpire::strategy::SizeLimiter>(
          name, rm.getAllocator("HOST"), 4 * 1024));
}

template <>
void StrategyTest<umpire::strategy::SlotPool>::SetUp()
{
  auto& rm = umpire::ResourceManager::getInstance();
  std::string name{"strategy_test_" + std::to_string(unique_strategy_id++)};

  m_allocator =
      new umpire::Allocator(rm.makeAllocator<umpire::strategy::SlotPool>(
          name, rm.getAllocator("HOST"), sizeof(double)));
}

template <>
void StrategyTest<umpire::strategy::MonotonicAllocationStrategy>::SetUp()
{
  auto& rm = umpire::ResourceManager::getInstance();
  std::string name{"strategy_test_" + std::to_string(unique_strategy_id++)};

  m_allocator = new umpire::Allocator(
      rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
          name, rm.getAllocator("HOST"), 4 * 1024));
}

using Strategies = ::testing::Types<
    umpire::strategy::AlignedAllocator,
#if defined(UMPIRE_ENABLE_CUDA)
    umpire::strategy::AllocationAdvisor,
#endif
    umpire::strategy::DynamicPool, umpire::strategy::DynamicPoolList,
    umpire::strategy::DynamicPoolMap, umpire::strategy::FixedPool,
    umpire::strategy::MixedPool, umpire::strategy::MonotonicAllocationStrategy,
    umpire::strategy::NamedAllocationStrategy, umpire::strategy::QuickPool,
    umpire::strategy::SizeLimiter, umpire::strategy::SlotPool,
    umpire::strategy::ThreadSafeAllocator>;

TYPED_TEST_SUITE(StrategyTest, Strategies, );

TYPED_TEST(StrategyTest, AllocateDeallocateBig)
{
  double* data = static_cast<double*>(
      this->m_allocator->allocate(this->m_big * sizeof(double)));

  ASSERT_NE(nullptr, data);

  this->m_allocator->deallocate(data);
}

TYPED_TEST(StrategyTest, MultipleAllocateDeallocate)
{
  const int number_of_allocations{8};
  std::vector<void*> allocations;

  for (int i{0}; i < number_of_allocations; ++i) {
    void* ptr = this->m_allocator->allocate(this->m_big * sizeof(double));
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
      this->m_allocator->allocate(this->m_nothing * sizeof(double)));

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
  const std::size_t size = this->m_big * sizeof(double);

  double* data = static_cast<double*>(this->m_allocator->allocate(size));

  ASSERT_EQ(size, this->m_allocator->getSize(data));

  this->m_allocator->deallocate(data);
}

TYPED_TEST(StrategyTest, GetById)
{
  auto& rm = umpire::ResourceManager::getInstance();

  int id = this->m_allocator->getId();
  ASSERT_GE(id, 0);

  auto allocator_by_id = rm.getAllocator(id);

  ASSERT_EQ(this->m_allocator->getAllocationStrategy(),
            allocator_by_id.getAllocationStrategy());

  ASSERT_THROW(rm.getAllocator(-25), umpire::util::Exception);
}

TYPED_TEST(StrategyTest, get_allocator_records)
{
  double* data = static_cast<double*>(
      this->m_allocator->allocate(this->m_big * sizeof(double)));

  auto records = umpire::get_allocator_records(*(this->m_allocator));

  ASSERT_EQ(records.size(), 1);

  this->m_allocator->deallocate(data);
}

TYPED_TEST(StrategyTest, getCurrentSize)
{
  ASSERT_EQ(this->m_allocator->getCurrentSize(), 0);

  void* data = this->m_allocator->allocate(this->m_big * sizeof(double));

  ASSERT_EQ(this->m_allocator->getCurrentSize(), this->m_big * sizeof(double));

  this->m_allocator->deallocate(data);
}

TYPED_TEST(StrategyTest, getActualSize)
{
  void* data = this->m_allocator->allocate(this->m_big * sizeof(double));
  ASSERT_GE(this->m_allocator->getActualSize(), this->m_big * sizeof(double));

  this->m_allocator->deallocate(data);
}

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

  ASSERT_NO_THROW({ alloc = allocator.allocate(100); });
  ASSERT_GE(allocator.getCurrentSize(), 100);
  ASSERT_EQ(allocator.getSize(alloc), 100);
  ASSERT_GE(allocator.getHighWatermark(), 100);
  ASSERT_NO_THROW({ allocator.deallocate(alloc); });

  std::size_t alloc_size = max_mem / 4;
  void* alloc1 = nullptr;
  void* alloc2 = nullptr;
  void* alloc3 = nullptr;

  // Hold a little of the first block we allocate
  ASSERT_NO_THROW({ alloc1 = allocator.allocate(1024); });
  ASSERT_NO_THROW({ alloc2 = allocator.allocate(1024); });
  ASSERT_NO_THROW({ allocator.deallocate(alloc1); });
  ASSERT_NO_THROW({ alloc3 = allocator.allocate(100); });
  ASSERT_NO_THROW({ allocator.deallocate(alloc2); });

  for (int i = 0; i < 16; ++i) {
    ASSERT_NO_THROW({ alloc1 = allocator.allocate(alloc_size); });
    ASSERT_NO_THROW({ allocator.deallocate(alloc1); });
    alloc_size += 1024;
  }

  ASSERT_NO_THROW({ allocator.deallocate(alloc3); });
}

TEST(MonotonicStrategy, Host)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator =
      rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
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

  auto allocator =
      rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
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

  auto allocator =
      rm.makeAllocator<umpire::strategy::MonotonicAllocationStrategy>(
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

  ASSERT_NO_THROW(auto read_only_alloc =
                      rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
                          "read_only_um", rm.getAllocator("UM"), "READ_MOSTLY");
                  UMPIRE_USE_VAR(read_only_alloc));

  ASSERT_ANY_THROW(
      auto failed_alloc = rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
          "read_only_um_nonsense_operator", rm.getAllocator("UM"), "FOOBAR");
      UMPIRE_USE_VAR(failed_alloc));
}

TEST(AllocationAdvisor, CreateWithId)
{
  auto& rm = umpire::ResourceManager::getInstance();

  const int device_id = 2;

  ASSERT_NO_THROW(auto read_only_alloc =
                      rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
                          "read_only_um_device_id", rm.getAllocator("UM"),
                          "READ_MOSTLY", device_id);
                  UMPIRE_USE_VAR(read_only_alloc));

  ASSERT_ANY_THROW(auto failed_alloc =
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

  auto read_only_alloc = rm.makeAllocator<umpire::strategy::AllocationAdvisor>(
      "preferred_location_host", um_allocator, "PREFERRED_LOCATION",
      host_allocator);

  ASSERT_NO_THROW({
    double* data =
        static_cast<double*>(read_only_alloc.allocate(1024 * sizeof(double)));
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
  ASSERT_GE(allocator.getActualSize(), data_size * 64);
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
  ASSERT_EQ(allocator.getSize(alloc[max_power - 1]), size / 4);
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

  for (std::size_t i = 0; i < N; i++) {
    threads.push_back(std::thread([=, &allocator, &thread_allocs] {
      for (int j = 0; j < N; ++j) {
        thread_allocs[i] = allocator.allocate(1024);
        ASSERT_NE(thread_allocs[i], nullptr);
        allocator.deallocate(thread_allocs[i]);
      }
      thread_allocs[i] = allocator.allocate(1024);
      ASSERT_NE(thread_allocs[i], nullptr);
    }));
  }

  for (auto& t : threads) {
    t.join();
  }

  for (auto alloc : thread_allocs) {
    ASSERT_NE(alloc, nullptr);
  }

  ASSERT_NO_THROW({
    for (auto alloc : thread_allocs) {
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
    const std::size_t size = 1024 * omp_get_thread_num();

    double* thread_data =
        static_cast<double*>(allocator.allocate(size * sizeof(double)));

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

  for (std::size_t i = 0; i < N; ++i) {
    threads.push_back(std::thread([=, &allocator, &thread_allocs] {
      for (int j = 0; j < N; ++j) {
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

  for (auto alloc : thread_allocs) {
    ASSERT_NE(alloc, nullptr);
  }

  ASSERT_NO_THROW({
    for (auto alloc : thread_allocs) {
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
    const std::size_t size = 1024 * omp_get_thread_num();

    double* thread_data =
        static_cast<double*>(allocator.allocate(size * sizeof(double)));

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

  EXPECT_THROW(
      {
        void* tmp_data = alloc.allocate(1024);
        UMPIRE_USE_VAR(tmp_data);
      },
      umpire::util::Exception);

  EXPECT_NO_THROW(alloc.deallocate(data));
}

#if defined(UMPIRE_ENABLE_NUMA)
TEST(NumaPolicyTest, EdgeCases)
{
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

TEST(NumaPolicyTest, Location)
{
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

static inline void test_alignment(uintptr_t p, unsigned int align)
{
  ASSERT_EQ(0, p % align);
  p &= align;
  ASSERT_TRUE(p == 0 || p == align);
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

  test_alignment(reinterpret_cast<uintptr_t>(d1), align);
  test_alignment(reinterpret_cast<uintptr_t>(d2), align);
  test_alignment(reinterpret_cast<uintptr_t>(d3), align);

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

  test_alignment(reinterpret_cast<uintptr_t>(d1), align);
  test_alignment(reinterpret_cast<uintptr_t>(d2), align);
  test_alignment(reinterpret_cast<uintptr_t>(d3), align);

  alloc.deallocate(d1);
  alloc.deallocate(d2);
  alloc.deallocate(d3);
}

TEST(AlignedAllocator, BadAlignment)
{
  auto& rm = umpire::ResourceManager::getInstance();

  EXPECT_THROW(
      {
        auto alloc = rm.makeAllocator<umpire::strategy::AlignedAllocator>(
            "aligned_allocator_6", rm.getAllocator("HOST"), 6);
        UMPIRE_USE_VAR(alloc);
      },
      umpire::util::Exception);

  EXPECT_THROW(
      {
        auto alloc = rm.makeAllocator<umpire::strategy::AlignedAllocator>(
            "aligned_allocator_11", rm.getAllocator("HOST"), 11);
        UMPIRE_USE_VAR(alloc);
      },
      umpire::util::Exception);

  EXPECT_THROW(
      {
        auto alloc = rm.makeAllocator<umpire::strategy::AlignedAllocator>(
            "aligned_allocator_0", rm.getAllocator("HOST"), 0);
        UMPIRE_USE_VAR(alloc);
      },
      umpire::util::Exception);
}
