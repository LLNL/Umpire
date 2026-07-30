//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/detail/registry.hpp"
#include "umpire/resource/host_memory.hpp"

#include <gtest/gtest.h>

#include <array>
#include <vector>

TEST(Registry, SingletonIdentity)
{
  auto& a = umpire::detail::registry::get();
  auto& b = umpire::detail::registry::get();
  EXPECT_EQ(&a, &b);
}

TEST(Registry, IdGeneration)
{
  auto& r = umpire::detail::registry::get();
  const int id0 = r.get_id();
  const int id1 = r.get_id();
  EXPECT_LT(id0, id1);
}

TEST(Registry, AllocationTracking)
{
  auto& r = umpire::detail::registry::get();

  int value = 0;
  umpire::allocation_record rec{&value, sizeof(value), nullptr};

  r.register_allocation(rec);
  EXPECT_TRUE(r.has_allocation(&value));

  auto found = r.find_allocation(&value);
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(found->ptr, &value);
  EXPECT_EQ(found->size, sizeof(value));

  r.remove_allocation(&value);
  EXPECT_FALSE(r.has_allocation(&value));
  EXPECT_FALSE(r.find_allocation(&value).has_value());
}

TEST(Registry, ContainingAllocationLookup)
{
  auto& r = umpire::detail::registry::get();

  // Two buffers registered out of address order to ensure the lookup does
  // not depend on insertion order.
  std::array<char, 64> low{};
  std::array<char, 64> high{};
  char* first = low.data() < high.data() ? low.data() : high.data();
  char* second = low.data() < high.data() ? high.data() : low.data();

  r.register_allocation({second, 64, nullptr});
  r.register_allocation({first, 64, nullptr});

  // Exact base pointer hit.
  auto found = r.find_containing_allocation(first);
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(found->ptr, static_cast<void*>(first));

  // Interior offset hit.
  found = r.find_containing_allocation(second + 63);
  ASSERT_TRUE(found.has_value());
  EXPECT_EQ(found->ptr, static_cast<void*>(second));

  // One-past-the-end must miss (unless it happens to be the base of the
  // other, non-adjacent buffer — the arrays are distinct locals so an
  // exact-adjacency collision would still resolve to the other record).
  if (first + 64 != second) {
    auto miss = r.find_containing_allocation(first + 64);
    if (miss.has_value()) {
      EXPECT_NE(miss->ptr, static_cast<void*>(first));
    }
  }

  r.remove_allocation(first);
  r.remove_allocation(second);

  EXPECT_FALSE(r.find_containing_allocation(first).has_value());
  EXPECT_FALSE(r.find_containing_allocation(second).has_value());
}

TEST(Registry, FindAllocationsByMemory)
{
  auto& r = umpire::detail::registry::get();
  auto& host = umpire::resource::host_memory<>::get();

  EXPECT_TRUE(r.find_allocations_by_memory(&host).empty());

  void* p1 = host.allocate(32);
  void* p2 = host.allocate(64);

  auto by_ptr = r.find_allocations_by_memory(&host);
  EXPECT_EQ(by_ptr.size(), 2u);

  auto by_id = r.find_allocations_by_memory(host.get_id());
  EXPECT_EQ(by_id.size(), 2u);
  std::size_t total = 0;
  for (const auto& record : by_id) {
    EXPECT_EQ(record.strategy, &host);
    total += record.size;
  }
  EXPECT_EQ(total, 96u);

  // Unknown id returns an empty result.
  EXPECT_TRUE(r.find_allocations_by_memory(-42).empty());

  host.deallocate(p1);
  host.deallocate(p2);

  EXPECT_TRUE(r.find_allocations_by_memory(&host).empty());
}
