//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

#include <algorithm>
#include <random>
#include <string>
#include <sstream>
#include <vector>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"


namespace {
  bool initialized{false};
  umpire::Allocator allocator;
  std::size_t largest_allocation_size{0};
  std::size_t initial_size{0};
  umpire::resource::HostSharedMemoryResource* shmem_resource{nullptr};

  class SharedMemoryTest : public ::testing::Test {
  public:
    using ArrayElement = int;

    const std::size_t m_segment_size{  512ULL * 1024ULL * 1024ULL };
    const std::size_t m_max_allocs{ 1024 };
    const std::size_t m_max_size{ m_segment_size / (m_max_allocs*sizeof(ArrayElement)) };

    virtual void SetUp()
    {
      if (!initialized) {
        auto& rm = umpire::ResourceManager::getInstance();
        auto traits{umpire::get_default_resource_traits("SHARED")};
        ASSERT_EQ(traits.scope, umpire::MemoryResourceTraits::shared_scope::node);
        ASSERT_EQ(traits.resource, umpire::MemoryResourceTraits::resource_type::shared);

        traits.size = m_segment_size;
        allocator = rm.makeResource("SHARED::node_allocator", traits);
        auto base_strategy = umpire::util::unwrap_allocator<umpire::strategy::AllocationStrategy>(allocator);
        shmem_resource = dynamic_cast<umpire::resource::HostSharedMemoryResource*>(base_strategy);

        initial_size = shmem_resource->getCurrentSize();
        largest_allocation_size = find_largest_allocation_size();
        initialized = true;
      }
    }

    virtual void TearDown()
    {
      ASSERT_EQ( initial_size, shmem_resource->getCurrentSize() );
      ASSERT_EQ( largest_allocation_size, find_largest_allocation_size() );
    }

    std::size_t find_largest_allocation_size()
    {
      //
      // The shared memory allocator will use its segment for both the data and
      // supporting metadata (segment header, storage for name, block descriptor
      // linked list, etc).  Consequently, the size used to create the allocator
      // will be too large.
      //
      // This test will continue allocating until it finds the largest size.  As
      // a unit test, this test will then (de)allocate that amount in a loop to
      // insure that there are no problems within the internal accounting in the
      // implementation.
      //
      std::size_t allocation_size{m_segment_size};
      void* ptr{nullptr};

      while (allocation_size != 0) {
        try {
          ptr = allocator.allocate("AllocLargest", allocation_size);
          break;
        }
        catch ( ... ) {
          allocation_size--;
          continue;
        }
      }

      allocator.deallocate(ptr);

      return allocation_size;
    }

    std::vector< std::pair< ArrayElement*, std::size_t > > allocs_until_full()
    {
      std::random_device rd;
      std::mt19937 gen{ rd() };
      std::uniform_int_distribution<std::size_t> distrib(1, m_max_size);
      std::vector< std::pair<ArrayElement*, std::size_t> > allocs;

      for ( std::size_t i = 0; i < m_max_allocs; ++i) {
        try {
          std::size_t num_elems{ distrib(gen) };
          std::stringstream name;
          name << "size_" << num_elems;
          void* ptr{ allocator.allocate(name.str(), num_elems * sizeof(ArrayElement)) };
          allocs.push_back( std::make_pair(reinterpret_cast<int*>(ptr), num_elems) );
        } catch (...) {
          break;
        }
      }

      return allocs;
    }

    void
    do_deallocations(std::vector<std::pair<ArrayElement*, std::size_t>>& allocs)
    {
      for ( auto& x : allocs ) {
        ASSERT_NO_THROW( allocator.deallocate(x.first); );
      }
    }

  };

  TEST_F(SharedMemoryTest, Construct)
  {
    SUCCEED();
  }

  TEST_F(SharedMemoryTest, AllocateTooMuch)
  {
    std::size_t allocation_size{m_segment_size+1};

    ASSERT_THROW( allocator.allocate("AllocTooMuch", allocation_size), umpire::util::Exception);
  }

  TEST_F(SharedMemoryTest, AllocateLargest)
  {
    ASSERT_EQ( largest_allocation_size, find_largest_allocation_size() );
    void* ptr{nullptr};

    for ( int loop{0}; loop < 1000; loop++ ) {
      ASSERT_NO_THROW( ptr = allocator.allocate("AllocLargest", largest_allocation_size); );
      ASSERT_NE(ptr, nullptr);
      ASSERT_NO_THROW( allocator.deallocate(ptr); );
    }
  }

  TEST_F(SharedMemoryTest, MixedAllocationSizes)
  {
    // Do deallocations in same order
    {
      auto allocs = allocs_until_full();
      ASSERT_EQ( allocs.size(), m_max_allocs );
      ASSERT_GT(shmem_resource->getCurrentSize(), initial_size);

      do_deallocations(allocs);
      ASSERT_EQ( allocs.size(), m_max_allocs );
      ASSERT_EQ(shmem_resource->getCurrentSize(), initial_size);
    }

    // Do deallocations in random order
    {
      auto allocs = allocs_until_full();
      ASSERT_EQ( allocs.size(), m_max_allocs );
      ASSERT_GT(shmem_resource->getCurrentSize(), initial_size);

      std::random_shuffle(allocs.begin(), allocs.end());

      do_deallocations(allocs);
      ASSERT_EQ( allocs.size(), m_max_allocs );
      ASSERT_EQ(shmem_resource->getCurrentSize(), initial_size);
    }
  }

  TEST_F(SharedMemoryTest, MixedAllocationSizesAndData)
  {
    auto allocs = allocs_until_full();
    ArrayElement counter{1};

    for ( auto& x : allocs ) {
      ArrayElement* p{ reinterpret_cast<ArrayElement*>(x.first) };
      std::size_t elems{x.second};
      for (std::size_t i{0}; i < elems; ++i) {
        p[i] = counter;
      }
      counter++;
    }

    // Now validate the data
    counter = 1;
    for ( auto& x : allocs ) {
      ArrayElement* p{ reinterpret_cast<ArrayElement*>(x.first) };
      std::size_t elems{x.second};
      for (std::size_t i{0}; i < elems; ++i) {
        ASSERT_EQ(p[i], counter);
      }
      counter++;
    }

    do_deallocations(allocs);
    ASSERT_EQ( allocs.size(), m_max_allocs );
    ASSERT_EQ(shmem_resource->getCurrentSize(), initial_size);
  }
}
