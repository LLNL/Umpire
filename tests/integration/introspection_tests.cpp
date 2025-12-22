//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"

TEST(IntrospectionTest, Overlaps)
{
#ifndef UMPIRE_ENABLE_HEADER_INTROSPECTION
  // This test validates map-based allocation tracking by manually registering allocations.
  // In header introspection mode, allocations are tracked via inline headers, not a central map.
  // Manually registered allocations don't have headers, so pointer_overlaps would read invalid memory.
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator allocator{rm.getAllocator("HOST")};
  umpire::strategy::AllocationStrategy* strategy{rm.getAllocator("HOST").getAllocationStrategy()};

  char* data{static_cast<char*>(allocator.allocate(4096))};

  {
    char* overlap_ptr = data + 17;
    auto overlap_record = umpire::util::AllocationRecord{overlap_ptr, 4096, strategy};
    rm.registerAllocation(overlap_ptr, overlap_record);

    ASSERT_TRUE(umpire::pointer_overlaps(data, overlap_ptr));

    rm.deregisterAllocation(overlap_ptr);
  }

  {
    char* overlap_ptr = data + 4095;
    auto overlap_record = umpire::util::AllocationRecord{overlap_ptr, 128, strategy};
    rm.registerAllocation(overlap_ptr, overlap_record);

    ASSERT_TRUE(umpire::pointer_overlaps(data, overlap_ptr));

    rm.deregisterAllocation(overlap_ptr);
  }

  {
    char* overlap_ptr = data + 4096;
    auto overlap_record = umpire::util::AllocationRecord{overlap_ptr, 128, strategy};
    rm.registerAllocation(overlap_ptr, overlap_record);

    ASSERT_FALSE(umpire::pointer_overlaps(data, overlap_ptr));

    rm.deregisterAllocation(overlap_ptr);
  }

  {
    char* overlap_ptr = data + 2048;
    auto overlap_record = umpire::util::AllocationRecord{overlap_ptr, 2047, strategy};
    rm.registerAllocation(overlap_ptr, overlap_record);

    ASSERT_FALSE(umpire::pointer_overlaps(data, overlap_ptr));
    rm.deregisterAllocation(overlap_ptr);
  }

  {
    char* overlap_ptr = data + 2048;
    auto overlap_record = umpire::util::AllocationRecord{overlap_ptr, 2048, strategy};
    rm.registerAllocation(overlap_ptr, overlap_record);

    ASSERT_FALSE(umpire::pointer_overlaps(data, overlap_ptr));
    rm.deregisterAllocation(overlap_ptr);
  }

  allocator.deallocate(data);
#else
  SUCCEED(); // Test not applicable in header introspection mode
#endif
}

TEST(IntrospectionTest, Contains)
{
#ifndef UMPIRE_ENABLE_HEADER_INTROSPECTION
  // This test validates map-based allocation tracking by manually registering allocations.
  // In header introspection mode, allocations are tracked via inline headers, not a central map.
  // Manually registered allocations don't have headers, so pointer_contains would read invalid memory.
  auto& rm = umpire::ResourceManager::getInstance();
  umpire::Allocator allocator{rm.getAllocator("HOST")};
  umpire::strategy::AllocationStrategy* strategy{rm.getAllocator("HOST").getAllocationStrategy()};

  char* data{static_cast<char*>(allocator.allocate(4096))};

  {
    char* contains_ptr = data + 17;
    auto contains_record = umpire::util::AllocationRecord{contains_ptr, 16, strategy};
    rm.registerAllocation(contains_ptr, contains_record);

    ASSERT_TRUE(umpire::pointer_contains(data, contains_ptr));

    rm.deregisterAllocation(contains_ptr);
  }

  allocator.deallocate(data);
#else
  SUCCEED(); // Test not applicable in header introspection mode
#endif
}

TEST(IntrospectionTest, RegisterNull)
{
  // This test validates that manually registering nullptr throws an error in both introspection modes.
  // Even though header introspection mode doesn't rely on manual registration as the primary tracking
  // mechanism, the registerAllocation API should still properly reject nullptr as invalid input.
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::strategy::AllocationStrategy* strategy{rm.getAllocator("HOST").getAllocationStrategy()};

  auto record = umpire::util::AllocationRecord{nullptr, 0, strategy};

  EXPECT_THROW(rm.registerAllocation(nullptr, record), umpire::runtime_error);
}

TEST(IntrospectionTest, ZeroByteAllocation)
{
  // This test documents the intentional behavior difference for zero-byte allocations between
  // map-based and header-based introspection modes.
  auto& rm = umpire::ResourceManager::getInstance();
  auto allocator = rm.getAllocator("HOST");

  void* ptr = allocator.allocate(0);

#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
  // Header mode: returns nullptr, not tracked
  ASSERT_EQ(ptr, nullptr);
  ASSERT_FALSE(rm.hasAllocator(ptr));

  // getSize(nullptr) should throw
  ASSERT_THROW(rm.getSize(ptr), umpire::runtime_error);

  // Deallocate nullptr should be safe (no-op)
  ASSERT_NO_THROW(allocator.deallocate(ptr));
#else
  // Map mode: returns unique non-null pointer, tracked
  ASSERT_NE(ptr, nullptr);
  ASSERT_TRUE(rm.hasAllocator(ptr));
  ASSERT_EQ(rm.getSize(ptr), 0);

  ASSERT_NO_THROW(allocator.deallocate(ptr));
#endif
}
