//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "camp/camp.hpp"
#include "gtest/gtest.h"
#include "test_helpers.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/DynamicPoolList.hpp"
#include "umpire/strategy/DynamicPoolMap.hpp"
#include "umpire/strategy/QuickPool.hpp"

template <>
struct tag_to_string<umpire::strategy::DynamicPoolList> {
  static constexpr const char* value = "DynamicPoolList";
};

template <>
struct tag_to_string<umpire::strategy::DynamicPoolMap> {
  static constexpr const char* value = "DynamicPoolMap";
};

template <>
struct tag_to_string<umpire::strategy::QuickPool> {
  static constexpr const char* value = "QuickPool";
};

using ResourceTypes = camp::list<host_resource_tag
#if defined(UMPIRE_ENABLE_DEVICE)
                                 ,
                                 device_resource_tag
#endif
#if defined(UMPIRE_ENABLE_UM)
                                 ,
                                 um_resource_tag
#endif
#if defined(UMPIRE_ENABLE_PINNED)
                                 ,
                                 pinned_resource_tag
#endif
#if defined(UMPIRE_ENABLE_FILE_RESOURCE)
                                 ,
                                 file_resource_tag
#endif
                                 >;

using PoolTypes =
    camp::list<
                  umpire::strategy::DynamicPoolList
                , umpire::strategy::DynamicPoolMap
                , umpire::strategy::QuickPool
              >;
using TestTypes = camp::cartesian_product<PoolTypes, ResourceTypes>;

using PoolTestTypes = Test<TestTypes>::Types;

template <typename PoolTuple>
class PrimaryPoolTest : public ::testing::Test {
 public:
  using Pool = typename camp::at<PoolTuple, camp::num<0>>::type;
  using ResourceType = typename camp::at<PoolTuple, camp::num<1>>::type;

  void SetUp() override
  {
    static int unique_counter{0};
    auto& rm = umpire::ResourceManager::getInstance();
    m_resource_name = std::string(tag_to_string<ResourceType>::value);

    m_pool_name = std::string{"pool_test"} + std::string{"_"} +
                  std::string{tag_to_string<Pool>::value} + std::string{"_"} +
                  std::string{m_resource_name} + std::string{"_"} +
                  std::to_string(unique_counter++);

    m_allocator = new umpire::Allocator(rm.makeAllocator<Pool>(
        m_pool_name, rm.getAllocator(m_resource_name), m_initial_pool_size,
        m_min_pool_growth_size, m_alignment));
  }

  void TearDown() override
  {
    delete m_allocator;
    m_allocator = nullptr;
  }

  umpire::Allocator* m_allocator;
  const std::size_t m_big{512};
  const std::size_t m_nothing{0};
  const std::size_t m_initial_pool_size{16 * 1024};
  const std::size_t m_min_pool_growth_size{1024};
  const std::size_t m_alignment{16};
  std::string m_pool_name;
  std::string m_resource_name;
};

TYPED_TEST_SUITE(PrimaryPoolTest, PoolTestTypes, );

TYPED_TEST(PrimaryPoolTest, Allocate)
{
  ASSERT_NO_THROW(
      this->m_allocator->deallocate(this->m_allocator->allocate(100)););
}

TYPED_TEST(PrimaryPoolTest, LazyFirstAllocation)
{
  ASSERT_EQ(this->m_allocator->getActualSize(), 0);

  ASSERT_NO_THROW(
      this->m_allocator->deallocate(this->m_allocator->allocate(100)););

  ASSERT_EQ(this->m_allocator->getActualSize(), this->m_initial_pool_size);
}

TYPED_TEST(PrimaryPoolTest, AllocateDeallocateBig)
{
  ASSERT_NO_THROW(this->m_allocator->deallocate(
      this->m_allocator->allocate(this->m_big * sizeof(double))););
}

TYPED_TEST(PrimaryPoolTest, Duplicate)
{
  using Pool = typename TestFixture::Pool;
  auto& rm = umpire::ResourceManager::getInstance();

  ASSERT_TRUE(rm.isAllocator(this->m_allocator->getName()));

  ASSERT_ANY_THROW(rm.makeAllocator<Pool>(
      this->m_allocator->getName(), rm.getAllocator(this->m_resource_name)));
}

//
// The getBlocksInPool statistic is expected to return the total number of
// blocks (allocated blocks to users + free blocks).  This test confirms
// this is consistent across all of the primary pools.
//
TYPED_TEST(PrimaryPoolTest, BlocksStatistic)
{
  using Pool = typename TestFixture::Pool;
  auto dynamic_pool = umpire::util::unwrap_allocator<Pool>(*this->m_allocator);

  ASSERT_NE(dynamic_pool, nullptr);

  const std::size_t iterations{16};
  void* allocs[iterations];

  // 2 Blocks (0 free, 2 allocated)
  for (int i{0}; i < 2; ++i) {
    ASSERT_NO_THROW(allocs[i] = this->m_allocator->allocate(
                        this->m_initial_pool_size / 2););
  }
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2);

  // 5 Blocks (1 Free, 4 allocated)
  for (int i{2}; i <= 3; ++i) {
    ASSERT_NO_THROW(allocs[i] = this->m_allocator->allocate(
                        this->m_min_pool_growth_size / 8););
  }

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 5);

  // 3 BLocks (1 Free, 2 allocated)
  for (int i{3}; i >= 2; --i) {
    ASSERT_NO_THROW(this->m_allocator->deallocate(allocs[i]););
  }

  //
  // The DynamicPoolMap does not recombine blocks dynamically, so we have
  // to (carefully) coalesce them here to have our total block count match
  // with the other pools.
  //
  if (std::is_same<Pool, umpire::strategy::DynamicPoolMap>::value) {
    dynamic_pool->coalesce();
  }
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3);

  // 1 BLocks (1 Free, 0 allocated) - Auto-collapsed
  for (int i{1}; i >= 0; --i) {
    ASSERT_NO_THROW(this->m_allocator->deallocate(allocs[i]););
  }
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);

  dynamic_pool->release();
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 0);
}

TYPED_TEST(PrimaryPoolTest, Sizes)
{
  void* data{nullptr};
  const std::size_t size{this->m_initial_pool_size - 1};

  ASSERT_NO_THROW(data = this->m_allocator->allocate(size););

  ASSERT_EQ(this->m_allocator->getSize(data), size);
  ASSERT_GE(this->m_allocator->getCurrentSize(), size);
  ASSERT_EQ(this->m_allocator->getHighWatermark(), size);
  ASSERT_EQ(this->m_allocator->getActualSize(), this->m_initial_pool_size);

  void* data2{nullptr};

  ASSERT_NO_THROW(data2 =
                      this->m_allocator->allocate(this->m_initial_pool_size););

  ASSERT_NO_THROW(this->m_allocator->deallocate(data););

  ASSERT_GE(this->m_allocator->getCurrentSize(), this->m_initial_pool_size);

  ASSERT_EQ(this->m_allocator->getHighWatermark(),
            this->m_initial_pool_size + size);

  ASSERT_GE(this->m_allocator->getActualSize(),
            this->m_initial_pool_size + this->m_min_pool_growth_size);

  ASSERT_EQ(this->m_allocator->getSize(data2), this->m_initial_pool_size);

  ASSERT_NO_THROW({ this->m_allocator->deallocate(data2); });
}

TYPED_TEST(PrimaryPoolTest, Alignment)
{
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<std::size_t> dist(1, this->m_big);
  std::vector<void*> allocations;
  const int num_allocations{1000};

  for (int i = 0; i < num_allocations; ++i) {
    const std::size_t size{dist(mt)};
    void* ptr{nullptr};

    ASSERT_NO_THROW({ ptr = this->m_allocator->allocate(size); });

    EXPECT_TRUE(0 ==
                (reinterpret_cast<std::ptrdiff_t>(ptr) % this->m_alignment))
        << "Allocation for size: " << size << " : " << ptr << " mod "
        << this->m_alignment << " = "
        << (reinterpret_cast<std::ptrdiff_t>(ptr) % this->m_alignment)
        << std::endl;

    ASSERT_TRUE(0 ==
                (reinterpret_cast<std::ptrdiff_t>(ptr) % this->m_alignment));

    allocations.push_back(ptr);
  }

  for (auto alloc : allocations) {
    ASSERT_NO_THROW(this->m_allocator->deallocate(alloc););
  }
}

TYPED_TEST(PrimaryPoolTest, Works)
{
  void* ptr_one{nullptr};
  void* ptr_two{nullptr};

  ASSERT_NO_THROW({
    ptr_one = this->m_allocator->allocate(62);
    ptr_two = this->m_allocator->allocate(1024);
    this->m_allocator->deallocate(ptr_two);
  });

  ASSERT_EQ(this->m_allocator->getCurrentSize(), 62);
  EXPECT_NO_THROW(this->m_allocator->release());

  ASSERT_LE(this->m_allocator->getActualSize(), this->m_initial_pool_size);

  ASSERT_NO_THROW(this->m_allocator->deallocate(ptr_one););

  ASSERT_EQ(this->m_allocator->getCurrentSize(), 0);

  EXPECT_NO_THROW(this->m_allocator->release());

  ASSERT_EQ(this->m_allocator->getCurrentSize(), 0);
  ASSERT_LE(this->m_allocator->getActualSize(), 0);
}

TYPED_TEST(PrimaryPoolTest, MissingBlocks)
{
  void* ptr_one{nullptr};
  void* ptr_two{nullptr};

  ASSERT_NO_THROW({
    ptr_one = this->m_allocator->allocate(128);
    ptr_two = this->m_allocator->allocate(44);
    this->m_allocator->deallocate(ptr_one);
    this->m_allocator->deallocate(ptr_two);
  });

  ASSERT_EQ(this->m_allocator->getCurrentSize(), 0);
  ASSERT_GE(this->m_allocator->getActualSize(), 0);

  this->m_allocator->release();

  ASSERT_EQ(this->m_allocator->getCurrentSize(), 0);
  ASSERT_EQ(this->m_allocator->getActualSize(), 0);
}

TYPED_TEST(PrimaryPoolTest, largestavailable)
{
  using Pool = typename TestFixture::Pool;
  const int num_allocs = 16;

  auto dynamic_pool = umpire::util::unwrap_allocator<Pool>(*this->m_allocator);

  ASSERT_NE(dynamic_pool, nullptr);

  ASSERT_NO_THROW({
    void* ptr{this->m_allocator->allocate(1024)};
    this->m_allocator->deallocate(ptr);
  });

  ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(),
            this->m_initial_pool_size);

  void* ptrs[num_allocs];

  for (int i{0}; i < num_allocs; ++i) {
    ASSERT_NO_THROW(ptrs[i] = this->m_allocator->allocate(1024););

    ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(),
              ((num_allocs - (i + 1)) * 1024));
  }

  for (int i{0}; i < num_allocs; i += 2) {
    ASSERT_NO_THROW(this->m_allocator->deallocate(ptrs[i]););
    ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(), 1024);
  }

  for (int i{1}; i < num_allocs; i += 2) {
    const int largest_block{((i + 2) < num_allocs) ? (i + 2) * 1024
                                                   : (i + 1) * 1024};
    ASSERT_NO_THROW(this->m_allocator->deallocate(ptrs[i]););
    ASSERT_EQ(dynamic_pool->getLargestAvailableBlock(), largest_block);
  }
}

TYPED_TEST(PrimaryPoolTest, coalesce)
{
  using Pool = typename TestFixture::Pool;

  auto dynamic_pool = umpire::util::unwrap_allocator<Pool>(*this->m_allocator);

  ASSERT_NE(dynamic_pool, nullptr);

  void* ptr_one{nullptr};
  void* ptr_two{nullptr};

  ASSERT_NO_THROW({
    ptr_one = this->m_allocator->allocate(1 + this->m_initial_pool_size -
                                          this->m_min_pool_growth_size);
    ptr_two = this->m_allocator->allocate(this->m_min_pool_growth_size);
  });

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3); // 1 Free, 2 Allocated
  ASSERT_NO_THROW(this->m_allocator->deallocate(ptr_two););

  dynamic_pool->coalesce();

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 3); // 2 Free, 1 Allocated

  ASSERT_NO_THROW(this->m_allocator->deallocate(ptr_one););

  dynamic_pool->coalesce();

  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1);

  ASSERT_EQ(this->m_allocator->getCurrentSize(), 0);
  ASSERT_EQ(dynamic_pool->getActualSize(),
            this->m_initial_pool_size + this->m_min_pool_growth_size);
  ASSERT_EQ(this->m_allocator->getHighWatermark(),
            1 + this->m_initial_pool_size);
}

TYPED_TEST(PrimaryPoolTest, heuristic_bounds)
{
  using Pool = typename TestFixture::Pool;
  EXPECT_THROW(
      {
        auto h = Pool::percent_releasable(-1);
        UMPIRE_USE_VAR(h);
      },
      umpire::util::Exception);

  EXPECT_THROW(
      {
        auto h = Pool::percent_releasable(101);
        UMPIRE_USE_VAR(h);
      },
      umpire::util::Exception);
}

TYPED_TEST(PrimaryPoolTest, heuristic_0_percent)
{
  const int initial_size{1024};
  const int subsequent_min_size{512};
  using Pool = typename TestFixture::Pool;
  auto& rm = umpire::ResourceManager::getInstance();

  auto h_fun = Pool::percent_releasable(0);
  auto alloc = rm.makeAllocator<Pool>(this->m_pool_name + std::string{"_0"},
                                      rm.getAllocator(this->m_resource_name),
                                      initial_size, subsequent_min_size,
                                      this->m_alignment, h_fun);

  auto dynamic_pool = umpire::util::unwrap_allocator<Pool>(alloc);

  ASSERT_NE(dynamic_pool, nullptr);

  //
  // After construction, we expect the pool to look like:
  // getCurrentSize     == 0
  // getActualSize      == 0
  //    block #1 (Whole Block: free(initial_size))
  //
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_LE(alloc.getActualSize(), initial_size);

  //
  // After alloc1=allocate(16), we expect the pool to look like:
  // getCurrentSize     == 16
  // getActualSize      == 1*initial_size
  //    block #1 (Partial Block: alloc1(16))
  //    block #2 (Partial Block: free(initial_size-16))
  //
  void* alloc1 = nullptr;
  ASSERT_NO_THROW({ alloc1 = alloc.allocate(16); });
  ASSERT_EQ(alloc.getCurrentSize(), 16);
  ASSERT_EQ(alloc.getActualSize(), initial_size);

  //
  // After alloc2=allocate(initial_size), we expect the pool to look like:
  // getCurrentSize     == 16+(1*initial_size)
  // getActualSize      == 2*initial_size
  //    block #1 (Partial Block: alloc1(16))
  //    block #2 (Partial Block: free(initial_size-16))
  //    block #3 (Whole Block: alloc2(initial_size))
  //
  void* alloc2 = nullptr;
  ASSERT_NO_THROW({ alloc2 = alloc.allocate(initial_size); });
  ASSERT_EQ(alloc.getCurrentSize(), 16 + initial_size);
  ASSERT_EQ(alloc.getActualSize(), 2 * initial_size);

  //
  // After alloc3=allocate(initial_size), we expect the pool to look like:
  // getCurrentSize     == 16+(2*initial_size)
  // getActualSize      == 3*initial_size
  //    block #1 (Partial Block: alloc1(16))
  //    block #2 (Partial Block: free(initial_size-16))
  //    block #3 (Whole Block: alloc2(initial_size))
  //    block #4 (Whole Block: alloc3(initial_size))
  //
  void* alloc3 = nullptr;
  ASSERT_NO_THROW({ alloc3 = alloc.allocate(initial_size); });
  ASSERT_EQ(alloc.getCurrentSize(), 16 + (2 * initial_size));
  ASSERT_EQ(alloc.getActualSize(), 3 * initial_size);

  //
  // After deallocate(alloc3), we expect the pool to look like:
  // getCurrentSize     == 16+(1*initial_size)
  // getActualSize      == 3*initial_size
  //    block #1 (Partial Block: alloc1(16))
  //    block #2 (Partial Block: free(initial_size-16))
  //    block #3 (Whole Block: alloc2(initial_size))
  //    block #4 (Whole Block: free(initial_size))
  //
  ASSERT_NO_THROW({ alloc.deallocate(alloc3); });
  ASSERT_EQ(alloc.getCurrentSize(), 16 + (1 * initial_size));
  ASSERT_EQ(alloc.getActualSize(), 3 * initial_size);

  //
  // After deallocate(alloc2), we expect the pool to look like:
  // getCurrentSize     == 16
  // getActualSize      == 3*initial_size
  //    block #1 (Partial Block: alloc1(16))
  //    block #2 (Partial Block: free(initial_size-16))
  //    block #3 (Whole Block: free(initial_size))
  //    block #4 (Whole Block: free(initial_size))
  //
  ASSERT_NO_THROW({ alloc.deallocate(alloc2); });
  ASSERT_EQ(alloc.getCurrentSize(), 16);
  ASSERT_EQ(alloc.getActualSize(), 3 * initial_size);

  //
  // After deallocate(alloc1), we expect the pool to look like:
  // getCurrentSize     == 0
  // getActualSize      == 3*initial_size
  //    block #1 (Partial Block: free(16))
  //    block #2 (Partial Block: free(initial_size-16))
  //    block #3 (Whole Block: free(initial_size))
  //    block #4 (Whole Block: free(initial_size))
  //
  ASSERT_NO_THROW({ alloc.deallocate(alloc1); });
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(alloc.getActualSize(), 3 * initial_size);

  //
  // After release, we expect the pool to look like:
  // getCurrentSize     == 0
  // getActualSize      == 0
  //
  EXPECT_NO_THROW(alloc.release());
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(alloc.getActualSize(), 0);

  //
  // final_alloc final_alloc=allocate(16), we expect the pool to look like:
  // getCurrentSize     == 16
  // getActualSize      == initial_size
  //    block #1 (Partial Block: final_alloc(16))
  //    block #2 (Partial Block: free(initial_size-16))
  //
  void* final_alloc = nullptr;
  ASSERT_NO_THROW({ final_alloc = alloc.allocate(16); });
  ASSERT_EQ(alloc.getCurrentSize(), 16);
  ASSERT_EQ(alloc.getActualSize(), initial_size);

  //
  // After deallocate(final_alloc), we expect the pool to look like:
  // getCurrentSize     == 0
  // getActualSize      == initial_size
  //    block #1 (Partial Block: free(16))
  //    block #2 (Partial Block: free(initial_size))
  //
  ASSERT_NO_THROW({ alloc.deallocate(final_alloc); });
  ASSERT_EQ(alloc.getCurrentSize(), 0);
  ASSERT_EQ(alloc.getActualSize(), initial_size);
}

TYPED_TEST(PrimaryPoolTest, heuristic_75_percent)
{
  const int initial_size{1024};
  const int subsequent_min_size{1024};
  using Pool = typename TestFixture::Pool;
  auto& rm = umpire::ResourceManager::getInstance();

  auto h_fun = Pool::percent_releasable(75);
  auto alloc = rm.makeAllocator<Pool>(this->m_pool_name + std::string{"_75"},
                                      rm.getAllocator(this->m_resource_name),
                                      initial_size, subsequent_min_size,
                                      this->m_alignment, h_fun);

  auto dynamic_pool = umpire::util::unwrap_allocator<Pool>(alloc);

  ASSERT_NE(dynamic_pool, nullptr);

  void* a[4];
  for (int i{0}; i < 4; ++i) {
    ASSERT_NO_THROW({ a[i] = alloc.allocate(1024); });
    ASSERT_EQ(alloc.getActualSize(), (1024 * (i + 1)));
    ASSERT_EQ(dynamic_pool->getBlocksInPool(), (i + 1));
    ASSERT_EQ(dynamic_pool->getReleasableSize(), 0);
  }

  // ASSERT_EQ(dynamic_pool->getBlocksInPool(), 4);
  ASSERT_NO_THROW({ alloc.deallocate(a[3]); }); // 25% releasable
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 4);
  ASSERT_NO_THROW({ alloc.deallocate(a[2]); }); // 50% releasable
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 4);
  ASSERT_NO_THROW({ alloc.deallocate(a[1]); });  // 75% releasable
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 2); // Collapse happened
  ASSERT_NO_THROW({ alloc.deallocate(a[0]); });  // 100% releasable
  ASSERT_EQ(dynamic_pool->getBlocksInPool(), 1); // Collapse happened
}

template <typename PoolTuple>
class PrimaryPoolTimingsTest : public ::testing::Test {
 public:
  using Pool = typename camp::at<PoolTuple, camp::num<0>>::type;
  using ResourceType = typename camp::at<PoolTuple, camp::num<1>>::type;

  void SetUp() override
  {
    m_resource_name = std::string(tag_to_string<ResourceType>::value);
    m_allocator = build_allocator<Pool, false>("pool_", 100);
    m_allocator_no_coalesce = build_allocator<Pool, false>("no_coalesce_pool", 0);
  }

  void TearDown() override
  {
    delete m_allocator;
    delete m_allocator_no_coalesce;
  }

  //
  // Returns the test duration in milliseconds
  //
  void run_test(umpire::Allocator* alloc, int& duration)
  {
    std::random_device rd;
    std::mt19937 g{rd()};

    auto start = std::chrono::steady_clock::now();

    for (std::size_t i{0}; i < m_max_allocs; ++i)
      ASSERT_NO_THROW( m_allocations[i] = alloc->allocate(1); );

    std::shuffle( m_allocations.begin(), m_allocations.end(), g );

    for (auto a : m_allocations )
      ASSERT_NO_THROW( alloc->deallocate(a); );

    auto end = std::chrono::steady_clock::now();

    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  }

  umpire::Allocator* m_allocator;
  umpire::Allocator* m_allocator_no_coalesce;

private:
  const std::size_t m_max_allocs{20000};
  std::vector<void*> m_allocations{m_max_allocs};
  std::string m_resource_name;

  template <typename P, bool use_introspection = true>
  umpire::Allocator* build_allocator(std::string name, int percentage)
  {
    static int unique_counter{0};
    const std::size_t initial_pool_size{512 * 1024 * 1024};
    const std::size_t min_pool_growth_size{1 * 1024 * 1024};
    const std::size_t alignment{16};
    const std::string pool_name{ name
                              + std::string{tag_to_string<Pool>::value}
                              + std::string{"_"}
                              + std::string{m_resource_name} + std::string{"_"}
                              + std::to_string(unique_counter++)};

    auto& rm = umpire::ResourceManager::getInstance();
    return new umpire::Allocator(
                rm.makeAllocator<P, use_introspection>(
                    pool_name
                  , rm.getAllocator(m_resource_name)
                  , initial_pool_size
                  , min_pool_growth_size
                  , alignment
                  , Pool::percent_releasable(percentage)
        )
     );
  }
};

TYPED_TEST_SUITE(PrimaryPoolTimingsTest, PoolTestTypes, );

TYPED_TEST(PrimaryPoolTimingsTest, TestCoalesceHeuristicTiming)
{
  //
  // Make sure that the time taken to run with the percent_releaseable(100)
  // heuristic is close to the same as the time taken to run
  // with the percent_releaseable(0) heuristic
  //
  const int max_delta{25};
  int ms_h_100{0};
  int ms_h_0{0};

  this->run_test(this->m_allocator, ms_h_100);
  this->run_test(this->m_allocator_no_coalesce, ms_h_0);

  int delta{ std::abs(ms_h_100 - ms_h_0) };

  if (delta >= max_delta) {
    std::cerr << "Difference between heuristic durations exceed maximum of: "
      << max_delta << std::endl
      << "Heuristic(100) Duration: " << ms_h_100
      << ", Heuristic(0) Duration: " << ms_h_0 << std::endl;
  }
  ASSERT_LT(delta, max_delta);
}
