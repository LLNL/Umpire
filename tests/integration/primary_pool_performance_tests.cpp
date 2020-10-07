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
    camp::list<umpire::strategy::DynamicPoolList,
               umpire::strategy::DynamicPoolMap, umpire::strategy::QuickPool>;
using TestTypes = camp::cartesian_product<PoolTypes, ResourceTypes>;

using PoolTestTypes = Test<TestTypes>::Types;

template <typename PoolTuple>
class PrimaryPoolTimingsTest : public ::testing::Test {
 public:
  using Pool = typename camp::at<PoolTuple, camp::num<0>>::type;
  using ResourceType = typename camp::at<PoolTuple, camp::num<1>>::type;

  void SetUp() override
  {
    m_resource_name = std::string(tag_to_string<ResourceType>::value);
    m_allocator = build_allocator<Pool, false>("pool_", 100);
    m_allocator_no_coalesce =
        build_allocator<Pool, false>("no_coalesce_pool", 0);
  }

  void TearDown() override
  {
    delete m_allocator;
    delete m_allocator_no_coalesce;
  }

  //
  // Returns the test duration in milliseconds
  //
  void run_test(umpire::Allocator* alloc, int64_t& duration)
  {
    std::random_device rd;
    std::mt19937 g{rd()};

    auto start = std::chrono::steady_clock::now();

    for (std::size_t i{0}; i < m_max_allocs; ++i)
      ASSERT_NO_THROW(m_allocations[i] = alloc->allocate(1););

    std::shuffle(m_allocations.begin(), m_allocations.end(), g);

    for (auto a : m_allocations)
      ASSERT_NO_THROW(alloc->deallocate(a););

    auto end = std::chrono::steady_clock::now();

    duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
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
    const std::string pool_name{
        name + std::string{tag_to_string<Pool>::value} + std::string{"_"} +
        std::string{m_resource_name} + std::string{"_"} +
        std::to_string(unique_counter++)};

    auto& rm = umpire::ResourceManager::getInstance();
    return new umpire::Allocator(rm.makeAllocator<P, use_introspection>(
        pool_name, rm.getAllocator(m_resource_name), initial_pool_size,
        min_pool_growth_size, alignment, Pool::percent_releasable(percentage)));
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
  int64_t ms_h_100{0};
  int64_t ms_h_0{0};

  this->run_test(this->m_allocator, ms_h_100);
  this->run_test(this->m_allocator_no_coalesce, ms_h_0);

  int64_t delta{std::abs(ms_h_100 - ms_h_0)};

  const int64_t max_delta{std::max((ms_h_100 / 4), INT64_C(25))};

  if (delta >= max_delta) {
    std::cerr << "Difference between heuristic durations exceed maximum of: "
              << max_delta << std::endl
              << "Heuristic(100) Duration: " << ms_h_100
              << ", Heuristic(0) Duration: " << ms_h_0 << std::endl;
  }
  ASSERT_LT(delta, max_delta);
}
