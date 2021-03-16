//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

#include <string>
#include <sstream>

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

  class SharedMemoryTest : public ::testing::Test {
    using ArrayElement = int;
    using ShapedArray = std::pair<ArrayElement*, std::size_t>;
  public:
    virtual void SetUp()
    {
      if (!initialized) {
        auto& rm = umpire::ResourceManager::getInstance();
        auto traits{umpire::get_default_resource_traits("SHARED")};
        ASSERT_EQ(traits.scope, umpire::MemoryResourceTraits::shared_scope::node);
        ASSERT_EQ(traits.resource, umpire::MemoryResourceTraits::resource_type::shared);

        traits.size = m_small;
        allocator = rm.makeResource("SHARED::node_allocator", traits);
        initialized = true;
        largest_allocation_size = find_largest_allocation_size();
      }
    }

    virtual void TearDown()
    {
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
      // This test will continue allocating until it finds the largest size.  As a
      // unit test, this test will then (de)allocate that amount in a loop to
      // insure that there are no problems within the internal accounting in the
      // implementation.
      //
      std::size_t allocation_size{m_small};
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

    const std::size_t m_small{  1ULL << 26ULL };  //  256 MiB
    // const std::size_t m_big  {  1ULL << 42ULL };  // 64 GiB
  };

  TEST_F(SharedMemoryTest, Construct)
  {
    SUCCEED();
  }

  TEST_F(SharedMemoryTest, AllocateTooMuch)
  {
    std::size_t allocation_size{m_small+1};

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
}
