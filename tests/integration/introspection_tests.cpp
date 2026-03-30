//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"

namespace {
class IntrospectionLevelGuard {
 public:
  explicit IntrospectionLevelGuard(umpire::ResourceManager& rm) : m_rm(rm), m_prev(rm.getIntrospectionLevel()) {}
  ~IntrospectionLevelGuard() { m_rm.setIntrospectionLevel(m_prev); }

  IntrospectionLevelGuard(const IntrospectionLevelGuard&) = delete;
  IntrospectionLevelGuard& operator=(const IntrospectionLevelGuard&) = delete;

 private:
  umpire::ResourceManager& m_rm;
  umpire::IntrospectionLevel m_prev;
};
} // namespace

TEST(IntrospectionTest, Overlaps)
{
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
}

TEST(IntrospectionTest, Contains)
{
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
}

TEST(IntrospectionTest, RegisterNull)
{
  auto& rm = umpire::ResourceManager::getInstance();

  umpire::strategy::AllocationStrategy* strategy{rm.getAllocator("HOST").getAllocationStrategy()};

  auto record = umpire::util::AllocationRecord{nullptr, 0, strategy};

  EXPECT_THROW(rm.registerAllocation(nullptr, record), umpire::runtime_error);
}

TEST(IntrospectionLevelTest, NamedAllocationRecordedByLevel)
{
  auto& rm = umpire::ResourceManager::getInstance();
  IntrospectionLevelGuard guard{rm};

  umpire::Allocator allocator{rm.getAllocator("HOST")};

  const std::string alloc_name{"my_named_alloc"};
  constexpr std::size_t size{64};

  rm.setIntrospectionLevel(umpire::IntrospectionLevel::Low);
  {
    void* p = allocator.allocate(alloc_name, size);
    ASSERT_TRUE(rm.hasAllocator(p));
    EXPECT_TRUE(rm.findAllocationRecord(p)->name.empty());
    allocator.deallocate(p);
  }

  rm.setIntrospectionLevel(umpire::IntrospectionLevel::Medium);
  {
    void* p = allocator.allocate(alloc_name, size);
    ASSERT_TRUE(rm.hasAllocator(p));
    EXPECT_EQ(rm.findAllocationRecord(p)->name, alloc_name);
    allocator.deallocate(p);
  }

  rm.setIntrospectionLevel(umpire::IntrospectionLevel::High);
  {
    void* p = allocator.allocate(alloc_name, size);
    ASSERT_TRUE(rm.hasAllocator(p));
    EXPECT_EQ(rm.findAllocationRecord(p)->name, alloc_name);
    allocator.deallocate(p);
  }
}
