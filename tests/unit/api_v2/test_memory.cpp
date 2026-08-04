//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/ResourceManager.hpp"
#include "umpire/detail/registry.hpp"
#include "umpire/error.hpp"
#include "umpire/memory.hpp"

#include <cstdlib>
#include <atomic>
#include <gtest/gtest.h>
#include <thread>
#include <string>

namespace {

class test_memory : public umpire::memory {
public:
  test_memory() : umpire::memory{"test"} { }
  explicit test_memory(const std::string& name) : umpire::memory{name} { }
  test_memory(const std::string& name, bool mirror_to_v1) : umpire::memory{name}
  {
    set_v1_mirroring(mirror_to_v1);
  }

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

TEST(memory, deallocate_through_wrong_owner_throws)
{
  test_memory owner{"owner"};
  test_memory other{"other"};

  void* ptr = owner.allocate(16);

  try {
    other.deallocate(ptr);
    FAIL() << "Expected unknown_allocation";
  } catch (const umpire::unknown_allocation& e) {
    const std::string what{e.what()};
    EXPECT_NE(what.find("owned by"), std::string::npos);
    EXPECT_NE(what.find("owner"), std::string::npos);
    EXPECT_NE(what.find("other"), std::string::npos);
  } catch (...) {
    FAIL() << "Expected umpire::unknown_allocation";
  }

  // The failed deallocation must not have disturbed the record or the
  // statistics of either object.
  EXPECT_TRUE(umpire::detail::registry::get().has_allocation(ptr));
  EXPECT_EQ(owner.get_current_size(), 16);
  EXPECT_EQ(other.get_current_size(), 0);

  owner.deallocate(ptr);
}

TEST(memory, destruction_with_active_allocations_warns_and_does_not_deallocate)
{
  void* leaked{nullptr};
  {
    test_memory mem{"leaky"};
    leaked = mem.allocate(32);
    // mem is destroyed here with one live allocation: the destructor must
    // warn (not throw) and must NOT deallocate the tracked allocation.
  }

  // The record intentionally survives the memory object's destruction.
  auto& registry = umpire::detail::registry::get();
  EXPECT_TRUE(registry.has_allocation(leaked));

  // Clean up the orphaned record and storage so other tests are unaffected.
  registry.remove_allocation(leaked);
  std::free(leaked);
}

TEST(memory, destruction_with_no_allocations_is_silent)
{
  {
    test_memory mem{"clean"};
    void* ptr = mem.allocate(8);
    mem.deallocate(ptr);
  }
  SUCCEED();
}

TEST(memory, registry_lookup_returns_stable_copy)
{
  test_memory mem;

  void* ptr = mem.allocate(16);
  auto record = umpire::detail::registry::get().find_allocation(ptr);

  ASSERT_TRUE(record.has_value());
  EXPECT_EQ(record->ptr, ptr);
  EXPECT_EQ(record->size, 16);
  EXPECT_EQ(record->strategy, &mem);

  mem.deallocate(ptr);
  EXPECT_EQ(record->ptr, ptr);
  EXPECT_EQ(record->size, 16);
  EXPECT_EQ(record->strategy, &mem);
}

TEST(memory, lookup_copy_survives_cross_thread_removal)
{
  test_memory mem;
  void* ptr = mem.allocate(24);

  std::optional<umpire::allocation_record> snapshot;
  std::atomic<bool> lookup_complete{false};

  std::thread reader([&]() {
    snapshot = umpire::detail::registry::get().find_allocation(ptr);
    lookup_complete.store(true, std::memory_order_release);
  });

  std::thread remover([&]() {
    while (!lookup_complete.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    mem.deallocate(ptr);
  });

  reader.join();
  remover.join();

  ASSERT_TRUE(snapshot.has_value());
  EXPECT_EQ(snapshot->ptr, ptr);
  EXPECT_EQ(snapshot->size, 24);
  EXPECT_EQ(snapshot->strategy, &mem);
}

// A memory object with v1 mirroring enabled (via the protected
// set_v1_mirroring() opt-in) registers each tracked allocation in v1's
// ResourceManager and deregisters it on deallocation. Since this object's
// name ("mirror_enabled") does not match any v1 allocator, the mirror
// falls back to the v1 "HOST" allocator's strategy because this object
// reports the host platform (see resolve_v1_mirror_strategy() in
// src/umpire/memory.cpp).
TEST(memory, v1_mirroring_enabled_registers_with_v1_resource_manager)
{
  auto& rm = umpire::ResourceManager::getInstance();
  test_memory mem{"mirror_enabled", true};

  void* ptr = mem.allocate(32);
  EXPECT_TRUE(rm.hasAllocator(ptr));

  auto* record = rm.findAllocationRecord(ptr);
  ASSERT_NE(record, nullptr);
  EXPECT_EQ(record->ptr, ptr);
  EXPECT_EQ(record->size, 32u);

  mem.deallocate(ptr);
  EXPECT_FALSE(rm.hasAllocator(ptr));
}

// A memory object without v1 mirroring enabled (the default) must NOT
// appear in v1's ResourceManager bookkeeping at all.
TEST(memory, v1_mirroring_disabled_does_not_register_with_v1_resource_manager)
{
  auto& rm = umpire::ResourceManager::getInstance();
  test_memory mem{"mirror_disabled", false};

  void* ptr = mem.allocate(32);
  EXPECT_FALSE(rm.hasAllocator(ptr));

  mem.deallocate(ptr);
  EXPECT_FALSE(rm.hasAllocator(ptr));
}

// The default constructor (no mirroring argument) must also leave v1's
// ResourceManager untouched, confirming the default-false behavior.
TEST(memory, default_construction_does_not_mirror_to_v1)
{
  auto& rm = umpire::ResourceManager::getInstance();
  test_memory mem{"mirror_default"};

  void* ptr = mem.allocate(16);
  EXPECT_FALSE(rm.hasAllocator(ptr));

  mem.deallocate(ptr);
}
