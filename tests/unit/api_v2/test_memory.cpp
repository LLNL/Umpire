//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/detail/registry.hpp"
#include "umpire/error.hpp"
#include "umpire/memory.hpp"

#include <cstdlib>
#include <gtest/gtest.h>
#include <string>

namespace {

class test_memory : public umpire::memory {
public:
  test_memory() : umpire::memory{"test"} { }

  void* allocate(std::size_t size) override
  {
    void* ptr{std::malloc(size)};
    track_allocation(ptr, size);
    return ptr;
  }

  void deallocate(void* ptr) override
  {
    untrack_allocation(ptr);
    std::free(ptr);
  }

  umpire::resource::Platform get_platform() const override { return umpire::resource::Platform::host; }
};

} // namespace

TEST(memory, registration_lifecycle)
{
  int id{-1};
  {
    test_memory mem;
    id = mem.get_id();
    EXPECT_EQ(umpire::detail::registry::get().find_allocator_by_id(id), &mem);
  }

  EXPECT_EQ(umpire::detail::registry::get().find_allocator_by_id(id), nullptr);
}

TEST(memory, statistics_and_highwatermark)
{
  test_memory mem;

  void* a = mem.allocate(8);
  EXPECT_EQ(mem.get_current_size(), 8);
  EXPECT_EQ(mem.get_highwatermark(), 8);

  void* b = mem.allocate(32);
  EXPECT_EQ(mem.get_current_size(), 40);
  EXPECT_EQ(mem.get_highwatermark(), 40);

  mem.deallocate(b);
  EXPECT_EQ(mem.get_current_size(), 8);
  EXPECT_EQ(mem.get_highwatermark(), 40);

  mem.deallocate(a);
  EXPECT_EQ(mem.get_current_size(), 0);
  EXPECT_EQ(mem.get_highwatermark(), 40);
}

TEST(memory, unknown_allocation_throws)
{
  test_memory mem;
  int i = 7;

  try {
    mem.deallocate(&i);
    FAIL() << "Expected unknown_allocation";
  } catch (const umpire::unknown_allocation& e) {
    EXPECT_NE(std::string{e.what()}.find("Attempted to deallocate unknown pointer"), std::string::npos);
    EXPECT_NE(std::string{e.what()}.find("0x"), std::string::npos);
  } catch (...) {
    FAIL() << "Expected umpire::unknown_allocation";
  }
}
