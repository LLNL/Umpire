//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-2025, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/allocation_strategy.hpp"
#include "umpire/memory.hpp"

#include <cstdlib>
#include <gtest/gtest.h>

namespace {

// Test memory implementation to serve as parent for strategy tests
class test_memory : public umpire::memory {
public:
  test_memory() : umpire::memory{"test_parent"} { }

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

  umpire::resource::Platform get_platform() const override {
    return umpire::resource::Platform::host;
  }
};

// Simple passthrough strategy for testing base functionality
class test_strategy : public umpire::allocation_strategy {
public:
  test_strategy(const std::string& name, umpire::memory* parent)
    : allocation_strategy(name, parent) {}

  void* allocate(std::size_t size) override {
    // Simple passthrough - delegates to parent
    return parent_->allocate(size);
  }

  void deallocate(void* ptr) override {
    // Simple passthrough - delegates to parent
    parent_->deallocate(ptr);
  }
};

// Nested strategy for composition testing
class nested_test_strategy : public umpire::allocation_strategy {
public:
  nested_test_strategy(const std::string& name, umpire::memory* parent)
    : allocation_strategy(name, parent) {}

  void* allocate(std::size_t size) override {
    return parent_->allocate(size);
  }

  void deallocate(void* ptr) override {
    parent_->deallocate(ptr);
  }
};

} // namespace

TEST(allocation_strategy, construct_with_valid_parent)
{
  test_memory parent;
  test_strategy strategy("test_strategy", &parent);

  EXPECT_EQ(strategy.get_parent(), &parent);
  EXPECT_EQ(strategy.get_name(), "test_strategy");
}

TEST(allocation_strategy, construct_with_null_parent_throws)
{
  EXPECT_THROW(
    test_strategy strategy("test_strategy", nullptr),
    std::invalid_argument
  );
}

TEST(allocation_strategy, get_platform_delegates_to_parent)
{
  test_memory parent;
  test_strategy strategy("test_strategy", &parent);

  EXPECT_EQ(strategy.get_platform(), parent.get_platform());
  EXPECT_EQ(strategy.get_platform(), umpire::resource::Platform::host);
}

TEST(allocation_strategy, allocate_delegates_to_parent)
{
  test_memory parent;
  test_strategy strategy("test_strategy", &parent);

  void* ptr = strategy.allocate(64);
  EXPECT_NE(ptr, nullptr);

  // Verify parent's statistics were updated
  EXPECT_EQ(parent.get_current_size(), 64);

  strategy.deallocate(ptr);
  EXPECT_EQ(parent.get_current_size(), 0);
}

TEST(allocation_strategy, get_parent_accessor)
{
  test_memory parent;
  test_strategy strategy("test_strategy", &parent);

  umpire::memory* retrieved_parent = strategy.get_parent();
  EXPECT_EQ(retrieved_parent, &parent);
}

TEST(allocation_strategy, composition_strategy_wrapping_strategy)
{
  test_memory resource;
  test_strategy inner_strategy("inner", &resource);
  nested_test_strategy outer_strategy("outer", &inner_strategy);

  // Verify composition chain
  EXPECT_EQ(outer_strategy.get_parent(), &inner_strategy);
  EXPECT_EQ(inner_strategy.get_parent(), &resource);

  // Platform should propagate through the chain
  EXPECT_EQ(outer_strategy.get_platform(), umpire::resource::Platform::host);

  // Allocate through composed strategies
  void* ptr = outer_strategy.allocate(128);
  EXPECT_NE(ptr, nullptr);

  // Resource should track the allocation
  EXPECT_EQ(resource.get_current_size(), 128);

  outer_strategy.deallocate(ptr);
  EXPECT_EQ(resource.get_current_size(), 0);
}

TEST(allocation_strategy, multiple_allocations_and_deallocations)
{
  test_memory parent;
  test_strategy strategy("test_strategy", &parent);

  void* ptr1 = strategy.allocate(32);
  void* ptr2 = strategy.allocate(64);
  void* ptr3 = strategy.allocate(128);

  EXPECT_EQ(parent.get_current_size(), 224);

  strategy.deallocate(ptr2);
  EXPECT_EQ(parent.get_current_size(), 160);

  strategy.deallocate(ptr1);
  EXPECT_EQ(parent.get_current_size(), 128);

  strategy.deallocate(ptr3);
  EXPECT_EQ(parent.get_current_size(), 0);
}
