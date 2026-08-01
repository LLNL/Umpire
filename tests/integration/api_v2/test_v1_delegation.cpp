//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

// This file is only built when UMPIRE_V1_DELEGATE_TO_V2 is enabled (see
// tests/integration/api_v2/CMakeLists.txt). It exercises the v1-strategy ->
// API v2 delegation adapter (umpire::strategy::detail::v1_backed_memory) via
// the three converted v1 strategies: SizeLimiter, ThreadSafeAllocator, and
// (indirectly, through registry visibility) any delegated strategy.

#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/detail/registry.hpp"
#include "umpire/error.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"

#include "gtest/gtest.h"

#include <atomic>
#include <cstddef>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string unique_allocator_name(const char* prefix)
{
  static std::atomic<int> counter{0};
  return std::string{prefix} + "_" + std::to_string(counter.fetch_add(1));
}

} // namespace

// A delegated SizeLimiter still throws the v1 exception type
// (umpire::out_of_memory_error) when the configured limit is exceeded, even
// though the underlying v2 strategy::size_limiter throws umpire::logic_error
// internally. See src/umpire/strategy/SizeLimiter.cpp for the catch/rethrow
// that preserves this v1 contract.
TEST(V1Delegation, DelegatedSizeLimiterEnforcesLimitWithV1ExceptionType)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto alloc = rm.makeAllocator<umpire::strategy::SizeLimiter>(unique_allocator_name("v1_delegate_size_limiter"),
                                                                rm.getAllocator("HOST"), 64);

  void* data = nullptr;
  EXPECT_NO_THROW(data = alloc.allocate(64));

  EXPECT_THROW(
      {
        void* tmp_data = alloc.allocate(1024);
        (void)tmp_data;
      },
      umpire::out_of_memory_error);

  EXPECT_NO_THROW(alloc.deallocate(data));
}

// A delegated ThreadSafeAllocator must remain safe for concurrent
// allocate()/deallocate() calls from multiple threads: the v2
// thread_safe<v1_backed_memory> delegate serializes calls through its own
// mutex, while the v1 counter path (m_current_size/m_allocation_count) is
// still updated per-call by AllocationStrategy::allocate_internal /
// deallocate_internal in the outer Allocator.inl call sequence.
TEST(V1Delegation, DelegatedThreadSafeAllocatorHandlesConcurrentAllocateDeallocate)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
      unique_allocator_name("v1_delegate_thread_safe"), rm.getAllocator("HOST"));

  constexpr int kThreads = 8;
  constexpr int kIterations = 64;

  std::vector<std::thread> threads;
  std::vector<void*> final_allocs(kThreads, nullptr);

  for (int i = 0; i < kThreads; ++i) {
    threads.emplace_back([i, &allocator, &final_allocs] {
      for (int j = 0; j < kIterations; ++j) {
        void* ptr = allocator.allocate(128);
        ASSERT_NE(ptr, nullptr);
        allocator.deallocate(ptr);
      }
      final_allocs[static_cast<std::size_t>(i)] = allocator.allocate(128);
      ASSERT_NE(final_allocs[static_cast<std::size_t>(i)], nullptr);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  for (auto* ptr : final_allocs) {
    ASSERT_NE(ptr, nullptr);
    allocator.deallocate(ptr);
  }

  EXPECT_EQ(allocator.getCurrentSize(), 0u);
  EXPECT_EQ(allocator.getAllocationCount(), 0u);
}

// Allocations made through a delegated strategy are visible in the v2
// registry (via find_allocations_by_memory, keyed on the strategy's
// v1_backed_memory bridge object) AND remain visible through the v1
// ResourceManager::findAllocationRecord interface, since v1's
// Allocator.inl / mixins::Inspector register/deregister every allocation
// against the outer v1 AllocationStrategy* regardless of delegation.
TEST(V1Delegation, DelegatedAllocationsVisibleInV1AndV2Registries)
{
  auto& rm = umpire::ResourceManager::getInstance();

  auto allocator = rm.makeAllocator<umpire::strategy::SizeLimiter>(
      unique_allocator_name("v1_delegate_registry_visibility"), rm.getAllocator("HOST"), 4096);

  void* ptr = allocator.allocate(256);
  ASSERT_NE(ptr, nullptr);

  // Visible via v1's ResourceManager.
  EXPECT_TRUE(rm.hasAllocator(ptr));
  auto* v1_record = rm.findAllocationRecord(ptr);
  ASSERT_NE(v1_record, nullptr);
  EXPECT_EQ(v1_record->ptr, ptr);
  EXPECT_EQ(v1_record->size, 256u);

  // Visible via the v2 registry too: the strategy's underlying
  // v1_backed_memory bridge object registered the allocation directly
  // against itself (see v1_backed_memory::allocate()), so it must show up
  // when looking up allocations for that specific memory* key. We can't get
  // at the private bridge pointer directly from the test, but we can
  // confirm the allocation is discoverable via find_allocation() (which
  // v1_backed_memory::deallocate() also relies on for size recovery).
  auto v2_record = umpire::detail::registry::get().find_allocation(ptr);
  ASSERT_TRUE(v2_record.has_value());
  EXPECT_EQ(v2_record->ptr, ptr);
  EXPECT_EQ(v2_record->size, 256u);
  ASSERT_NE(v2_record->strategy, nullptr);

  auto by_memory = umpire::detail::registry::get().find_allocations_by_memory(v2_record->strategy);
  EXPECT_FALSE(by_memory.empty());
  bool found = false;
  for (const auto& record : by_memory) {
    if (record.ptr == ptr && record.size == 256u) {
      found = true;
      break;
    }
  }
  EXPECT_TRUE(found);

  allocator.deallocate(ptr);

  // Deallocation removes the record from both registries.
  EXPECT_FALSE(rm.hasAllocator(ptr));
  EXPECT_FALSE(umpire::detail::registry::get().find_allocation(ptr).has_value());
}
