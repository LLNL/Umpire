//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

#include "umpire/event/event.hpp"
#include "umpire/event/recorder_chain.hpp"

#include <stdexcept>

using namespace umpire::event;

// Mock event store for testing
class mock_event_store : public event_store {
 public:
  mock_event_store() : m_insert_count(0), m_should_throw(false) {}

  void insert(const event& e) override
  {
    if (m_should_throw) {
      throw std::runtime_error("Mock store error");
    }
    m_insert_count++;
    m_last_event = e;
  }

  void insert(const allocate& e) override
  {
    if (m_should_throw) {
      throw std::runtime_error("Mock store error");
    }
    m_insert_count++;
    m_last_allocate = e;
  }

  void insert(const named_allocate& e) override
  {
    if (m_should_throw) {
      throw std::runtime_error("Mock store error");
    }
    m_insert_count++;
    m_last_named_allocate = e;
  }

  void insert(const allocate_resource& e) override
  {
    if (m_should_throw) {
      throw std::runtime_error("Mock store error");
    }
    m_insert_count++;
    m_last_allocate_resource = e;
  }

  void insert(const deallocate& e) override
  {
    if (m_should_throw) {
      throw std::runtime_error("Mock store error");
    }
    m_insert_count++;
    m_last_deallocate = e;
  }

  void insert(const deallocate_resource& e) override
  {
    if (m_should_throw) {
      throw std::runtime_error("Mock store error");
    }
    m_insert_count++;
    m_last_deallocate_resource = e;
  }

  std::vector<event> get_events() override { return {}; }

  int m_insert_count;
  bool m_should_throw;
  event m_last_event;
  allocate m_last_allocate;
  named_allocate m_last_named_allocate;
  allocate_resource m_last_allocate_resource;
  deallocate m_last_deallocate;
  deallocate_resource m_last_deallocate_resource;
};

TEST(RecorderChainTest, EmptyChain)
{
  recorder_chain chain;

  // Inserting into empty chain should not crash
  event e;
  e.name = "test_event";
  chain.insert(e);

  // Should return empty vector
  EXPECT_TRUE(chain.get_events().empty());
}

TEST(RecorderChainTest, SingleStore)
{
  recorder_chain chain;
  mock_event_store store;

  chain.add_store(&store);

  event e;
  e.name = "test_event";
  chain.insert(e);

  EXPECT_EQ(store.m_insert_count, 1);
  EXPECT_EQ(store.m_last_event.name, "test_event");
}

TEST(RecorderChainTest, MultipleStores)
{
  recorder_chain chain;
  mock_event_store store1;
  mock_event_store store2;
  mock_event_store store3;

  chain.add_store(&store1);
  chain.add_store(&store2);
  chain.add_store(&store3);

  event e;
  e.name = "test_event";
  chain.insert(e);

  // All stores should receive the event
  EXPECT_EQ(store1.m_insert_count, 1);
  EXPECT_EQ(store2.m_insert_count, 1);
  EXPECT_EQ(store3.m_insert_count, 1);
}

TEST(RecorderChainTest, AllocateEvent)
{
  recorder_chain chain;
  mock_event_store store;

  chain.add_store(&store);

  allocate a;
  a.size = 1024;
  a.ptr = reinterpret_cast<void*>(0x1000);
  a.ref = reinterpret_cast<void*>(0x100);

  chain.insert(a);

  EXPECT_EQ(store.m_insert_count, 1);
  EXPECT_EQ(store.m_last_allocate.size, 1024);
}

TEST(RecorderChainTest, DeallocateEvent)
{
  recorder_chain chain;
  mock_event_store store;

  chain.add_store(&store);

  deallocate d;
  d.ptr = reinterpret_cast<void*>(0x1000);
  d.ref = reinterpret_cast<void*>(0x100);

  chain.insert(d);

  EXPECT_EQ(store.m_insert_count, 1);
  EXPECT_EQ(store.m_last_deallocate.ptr, reinterpret_cast<void*>(0x1000));
}

TEST(RecorderChainTest, ErrorIsolation)
{
  recorder_chain chain;
  mock_event_store store1;
  mock_event_store store2;
  mock_event_store store3;

  chain.add_store(&store1);
  chain.add_store(&store2);
  chain.add_store(&store3);

  // Make store2 throw
  store2.m_should_throw = true;

  event e;
  e.name = "test_event";

  // Should not crash even though store2 throws
  chain.insert(e);

  // Store1 and store3 should still receive the event
  EXPECT_EQ(store1.m_insert_count, 1);
  EXPECT_EQ(store2.m_insert_count, 0);  // Threw before counting
  EXPECT_EQ(store3.m_insert_count, 1);
}

TEST(RecorderChainTest, AllStoresThrow)
{
  recorder_chain chain;
  mock_event_store store1;
  mock_event_store store2;

  chain.add_store(&store1);
  chain.add_store(&store2);

  store1.m_should_throw = true;
  store2.m_should_throw = true;

  event e;
  e.name = "test_event";

  // Should not crash even though all stores throw
  chain.insert(e);

  EXPECT_EQ(store1.m_insert_count, 0);
  EXPECT_EQ(store2.m_insert_count, 0);
}

TEST(RecorderChainTest, AddNullStore)
{
  recorder_chain chain;

  // Adding null should not crash
  chain.add_store(nullptr);

  event e;
  e.name = "test_event";

  // Should not crash
  chain.insert(e);
}

TEST(RecorderChainTest, NamedAllocateEvent)
{
  recorder_chain chain;
  mock_event_store store;

  chain.add_store(&store);

  named_allocate na;
  na.size = 2048;
  na.ptr = reinterpret_cast<void*>(0x2000);
  na.ref = reinterpret_cast<void*>(0x200);
  na.name = "my_allocation";

  chain.insert(na);

  EXPECT_EQ(store.m_insert_count, 1);
  EXPECT_EQ(store.m_last_named_allocate.size, 2048);
  EXPECT_EQ(store.m_last_named_allocate.name, "my_allocation");
}

TEST(RecorderChainTest, AllocateResourceEvent)
{
  recorder_chain chain;
  mock_event_store store;

  chain.add_store(&store);

  allocate_resource ar;
  ar.size = 4096;
  ar.ptr = reinterpret_cast<void*>(0x3000);
  ar.ref = reinterpret_cast<void*>(0x300);
  ar.res = "cuda";

  chain.insert(ar);

  EXPECT_EQ(store.m_insert_count, 1);
  EXPECT_EQ(store.m_last_allocate_resource.size, 4096);
  EXPECT_EQ(store.m_last_allocate_resource.res, "cuda");
}

TEST(RecorderChainTest, DeallocateResourceEvent)
{
  recorder_chain chain;
  mock_event_store store;

  chain.add_store(&store);

  deallocate_resource dr;
  dr.ptr = reinterpret_cast<void*>(0x4000);
  dr.ref = reinterpret_cast<void*>(0x400);
  dr.res = "hip";

  chain.insert(dr);

  EXPECT_EQ(store.m_insert_count, 1);
  EXPECT_EQ(store.m_last_deallocate_resource.ptr, reinterpret_cast<void*>(0x4000));
  EXPECT_EQ(store.m_last_deallocate_resource.res, "hip");
}
