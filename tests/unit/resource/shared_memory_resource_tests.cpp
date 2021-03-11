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

std::size_t unique_id{0};

static std::string unique_name(std::string prefix) {
  std::stringstream ss;

  ss << prefix << unique_id++;
  return ss.str();
}

const std::size_t SmallSizedAllocator{  1ULL << 18ULL };  //  1 MiB
// const std::size_t MediumSizedAllocator{ 1ULL << 36ULL };  //  1 GiB
// const std::size_t LargeSizedAllocator{  1ULL << 42ULL };  // 64 GiB

TEST(SharedMemory, DefaultTraits)
{
  auto traits{umpire::get_default_resource_traits("SHARED")};
  ASSERT_EQ(traits.scope, umpire::MemoryResourceTraits::shared_scope::node);
  ASSERT_EQ(traits.resource, umpire::MemoryResourceTraits::resource_type::shared);
}

TEST(SharedMemory, MakeResource)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto traits{umpire::get_default_resource_traits("SHARED")};
  ASSERT_EQ(traits.scope, umpire::MemoryResourceTraits::shared_scope::node);
  ASSERT_EQ(traits.resource, umpire::MemoryResourceTraits::resource_type::shared);

  traits.size = SmallSizedAllocator;
  rm.makeResource(unique_name("SHARED::node_allocator"), traits);
}

TEST(SharedMemory, AllocateTooMuch)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto traits{umpire::get_default_resource_traits("SHARED")};
  ASSERT_EQ(traits.scope, umpire::MemoryResourceTraits::shared_scope::node);
  ASSERT_EQ(traits.resource, umpire::MemoryResourceTraits::resource_type::shared);

  traits.size = SmallSizedAllocator;
  auto alloc { rm.makeResource(unique_name("SHARED::node_allocator"), traits) };

  std::size_t allocation_size{SmallSizedAllocator+1};

  ASSERT_THROW( alloc.allocate("AllocTooMuch", allocation_size), umpire::util::Exception);
}

TEST(SharedMemory, AllocateLargest)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto traits{umpire::get_default_resource_traits("SHARED")};
  ASSERT_EQ(traits.scope, umpire::MemoryResourceTraits::shared_scope::node);
  ASSERT_EQ(traits.resource, umpire::MemoryResourceTraits::resource_type::shared);

  traits.size = SmallSizedAllocator;
  auto alloc { rm.makeResource(unique_name("SHARED::node_allocator"), traits) };

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
  std::size_t allocation_size{SmallSizedAllocator};
  void* ptr{nullptr};

  while (allocation_size != 0) {
    try {
      ptr = alloc.allocate("AllocLargest", allocation_size);
      break;
    }
    catch ( ... ) {
      allocation_size--;
      continue;
    }
  }

  ASSERT_NE(allocation_size, 0);
  ASSERT_NE(ptr, nullptr);
  ASSERT_NO_THROW( alloc.deallocate(ptr); );

  for ( int loop{0}; loop < 1000; loop++ ) {
    ASSERT_NO_THROW( ptr = alloc.allocate("AllocLargest", allocation_size); );
    ASSERT_NE(ptr, nullptr);
    ASSERT_NO_THROW( alloc.deallocate(ptr); );
  }
}
