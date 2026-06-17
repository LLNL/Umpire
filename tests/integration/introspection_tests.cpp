//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/QuickPool.hpp"

namespace {
umpire::IntrospectionLevel getCurrentLevel() {
  return umpire::ResourceManager::getInstance().getIntrospectionLevel();
}
} // namespace

TEST(IntrospectionTest, Overlaps)
{
  auto& rm = umpire::ResourceManager::getInstance();

  if (getCurrentLevel() != umpire::IntrospectionLevel::On) {
    GTEST_SKIP() << "Overlaps test requires introspection level 'on'";
  }

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

  if (getCurrentLevel() != umpire::IntrospectionLevel::On) {
    GTEST_SKIP() << "Contains test requires introspection level 'on'";
  }

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

  if (getCurrentLevel() != umpire::IntrospectionLevel::On) {
    GTEST_SKIP() << "RegisterNull test requires introspection level 'on'";
  }

  umpire::strategy::AllocationStrategy* strategy{rm.getAllocator("HOST").getAllocationStrategy()};

  auto record = umpire::util::AllocationRecord{nullptr, 0, strategy};

  EXPECT_THROW(rm.registerAllocation(nullptr, record), umpire::runtime_error);
}

TEST(IntrospectionLevelTest, OnTracksNamedAllocationMetadata)
{
  auto& rm = umpire::ResourceManager::getInstance();

  if (getCurrentLevel() != umpire::IntrospectionLevel::On) {
    GTEST_SKIP() << "OnTracksNamedAllocationMetadata test requires introspection level 'on'";
  }

  umpire::Allocator allocator{rm.getAllocator("HOST")};

  const std::string alloc_name{"my_named_alloc"};
  constexpr std::size_t size{64};

  void* p = allocator.allocate(alloc_name, size);
  ASSERT_TRUE(rm.hasAllocator(p));
  EXPECT_NO_THROW(rm.getAllocator(p));
  EXPECT_EQ(rm.getSize(p), size);
  EXPECT_EQ(rm.findAllocationRecord(p)->name, alloc_name);
  allocator.deallocate(p);
}

TEST(IntrospectionLevelTest, AllocationQueries)
{
  auto& rm = umpire::ResourceManager::getInstance();
  const auto level = getCurrentLevel();

  umpire::Allocator allocator{rm.getAllocator("HOST")};
  const std::string alloc_name{"my_named_alloc"};
  constexpr std::size_t size{64};

  void* p = allocator.allocate(alloc_name, size);

  if (level == umpire::IntrospectionLevel::Off) {
    // Off mode: no introspection available
    EXPECT_FALSE(rm.hasAllocator(p));
    EXPECT_THROW(rm.findAllocationRecord(p), umpire::runtime_error);
    EXPECT_THROW(rm.getAllocator(p), umpire::runtime_error);
    EXPECT_THROW(rm.getSize(p), umpire::runtime_error);
    EXPECT_THROW(umpire::get_allocator_records(allocator), umpire::runtime_error);

  } else if (level == umpire::IntrospectionLevel::Basic) {
    // Basic mode: API inference, no metadata
    ASSERT_TRUE(rm.hasAllocator(p));
    EXPECT_TRUE(rm.hasAllocator(static_cast<char*>(p) + 1));  // API can't track offsets
    EXPECT_THROW(rm.findAllocationRecord(p), umpire::runtime_error);
    EXPECT_NO_THROW(rm.getAllocator(p));  // Works via API inference
    EXPECT_THROW(rm.getSize(p), umpire::runtime_error);
    EXPECT_THROW(umpire::get_allocator_records(allocator), umpire::runtime_error);

  } else {  // IntrospectionLevel::On
    // On mode: full tracking with metadata
    ASSERT_TRUE(rm.hasAllocator(p));
    EXPECT_NO_THROW(rm.getAllocator(p));
    EXPECT_EQ(rm.getSize(p), size);
    EXPECT_EQ(rm.findAllocationRecord(p)->name, alloc_name);
    EXPECT_NO_THROW(umpire::get_allocator_records(allocator));
  }

  allocator.deallocate(p);
}

TEST(IntrospectionLevelTest, EdgeCaseStackPointer)
{
  auto& rm = umpire::ResourceManager::getInstance();
  const auto level = getCurrentLevel();

  int stack_var = 42;
  void* stack_ptr = &stack_var;

  if (level == umpire::IntrospectionLevel::Off) {
    EXPECT_FALSE(rm.hasAllocator(stack_ptr));

  } else if (level == umpire::IntrospectionLevel::Basic) {
    // May return true (infers HOST) or false (API fails)
    // Either is acceptable for non-Umpire pointer
    bool has_alloc = rm.hasAllocator(stack_ptr);
    (void)has_alloc; // Should not crash

  } else {  // On
    EXPECT_FALSE(rm.hasAllocator(stack_ptr));
    EXPECT_THROW(rm.getAllocator(stack_ptr), umpire::runtime_error);
  }
}

TEST(IntrospectionLevelTest, EdgeCaseFreedPointer)
{
  auto& rm = umpire::ResourceManager::getInstance();
  const auto level = getCurrentLevel();

  if (level == umpire::IntrospectionLevel::Off) {
    GTEST_SKIP() << "EdgeCaseFreedPointer test requires introspection (basic or on)";
  }

  umpire::Allocator allocator{rm.getAllocator("HOST")};
  void* ptr = allocator.allocate(256);

  if (level == umpire::IntrospectionLevel::On) {
    EXPECT_TRUE(rm.hasAllocator(ptr));
    allocator.deallocate(ptr);
    // After deallocation, should not be tracked
    EXPECT_FALSE(rm.hasAllocator(ptr));
    EXPECT_THROW(rm.getAllocator(ptr), umpire::runtime_error);

  } else {  // Basic
    allocator.deallocate(ptr);
    // Behavior after free is undefined in Basic mode but shouldn't crash
    bool has_alloc = rm.hasAllocator(ptr);
    (void)has_alloc;
  }
}

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
TEST(IntrospectionLevelTest, AsyncCopy)
{
  auto& rm = umpire::ResourceManager::getInstance();
  const auto level = getCurrentLevel();

  if (level != umpire::IntrospectionLevel::Basic) {
    GTEST_SKIP() << "AsyncCopy test specific to Basic mode";
  }

  umpire::Allocator device_alloc{rm.getAllocator("DEVICE")};
  umpire::Allocator host_alloc{rm.getAllocator("HOST")};

  void* device_ptr = device_alloc.allocate(256);
  void* host_ptr = host_alloc.allocate(256);

#if defined(UMPIRE_ENABLE_CUDA)
  auto ctx = camp::resources::Cuda();
#elif defined(UMPIRE_ENABLE_HIP)
  auto ctx = camp::resources::Hip();
#endif

  // Async copy should work (unsafe, no validation)
  EXPECT_NO_THROW(rm.copy(device_ptr, host_ptr, ctx, 256));
  ctx.wait();

  // Async copy with size=0 should throw
  EXPECT_THROW(rm.copy(device_ptr, host_ptr, ctx, 0), umpire::runtime_error);

  device_alloc.deallocate(device_ptr);
  host_alloc.deallocate(host_ptr);
}
#endif

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
TEST(IntrospectionLevelTest, AsyncMemset)
{
  auto& rm = umpire::ResourceManager::getInstance();
  const auto level = getCurrentLevel();

  if (level != umpire::IntrospectionLevel::Basic) {
    GTEST_SKIP() << "AsyncMemset test specific to Basic mode";
  }

  umpire::Allocator device_alloc{rm.getAllocator("DEVICE")};
  void* device_ptr = device_alloc.allocate(256);

#if defined(UMPIRE_ENABLE_CUDA)
  auto ctx = camp::resources::Cuda();
#elif defined(UMPIRE_ENABLE_HIP)
  auto ctx = camp::resources::Hip();
#endif

  // Async memset should work
  EXPECT_NO_THROW(rm.memset(device_ptr, 0, ctx, 256));
  ctx.wait();

  // Async memset with length=0 should throw
  EXPECT_THROW(rm.memset(device_ptr, 0, ctx, 0), umpire::runtime_error);

  device_alloc.deallocate(device_ptr);
}
#endif

TEST(IntrospectionLevelTest, PoolAllocatorIdentification)
{
  auto& rm = umpire::ResourceManager::getInstance();
  const auto level = getCurrentLevel();

  if (level == umpire::IntrospectionLevel::Off) {
    GTEST_SKIP() << "PoolAllocatorIdentification requires introspection (basic or on)";
  }

  umpire::Allocator host_alloc{rm.getAllocator("HOST")};
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("TestPool", host_alloc);
  void* pool_ptr = pool.allocate(256);

  auto retrieved = rm.getAllocator(pool_ptr);

  if (level == umpire::IntrospectionLevel::On) {
    // On mode returns the specific pool
    EXPECT_EQ(retrieved.getName(), "TestPool");
  } else {  // Basic
    // Basic mode returns backing resource
    EXPECT_EQ(retrieved.getName(), "HOST");
  }

  pool.deallocate(pool_ptr);
}

TEST(IntrospectionLevelTest, ErrorMessageQuality)
{
  auto& rm = umpire::ResourceManager::getInstance();
  const auto level = getCurrentLevel();

  umpire::Allocator allocator{rm.getAllocator("HOST")};

  if (level == umpire::IntrospectionLevel::Basic) {
    // In Basic mode, test that operations requiring full introspection give clear error
    void* p1 = allocator.allocate(256);

    try {
      rm.copy(p1, p1, 0);
      FAIL() << "Expected runtime_error";
    } catch (const umpire::runtime_error& e) {
      std::string msg = e.what();
      EXPECT_TRUE(msg.find("introspection") != std::string::npos &&
                  (msg.find("on") != std::string::npos || msg.find("On") != std::string::npos))
        << "Error message should mention introspection level: " << msg;
    }

    allocator.deallocate(p1);

  } else if (level == umpire::IntrospectionLevel::Off) {
    // Test Off mode operation error
    void* p = allocator.allocate(256);

    try {
      rm.getAllocator(p);
      FAIL() << "Expected runtime_error";
    } catch (const umpire::runtime_error& e) {
      std::string msg = e.what();
      EXPECT_TRUE(msg.find("introspection") != std::string::npos)
        << "Error message should mention introspection: " << msg;
    }

    allocator.deallocate(p);

  } else {  // On mode
    // Test On mode size=0 auto-sizing error
    void* p1 = allocator.allocate(256);
    void* p2 = allocator.allocate(256);

    // Size=0 with On mode should work (auto-sizing from allocation record)
    EXPECT_NO_THROW(rm.copy(p2, p1, 0));

    allocator.deallocate(p1);
    allocator.deallocate(p2);
  }
}

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
TEST(IntrospectionLevelTest, MultiGPU)
{
  auto& rm = umpire::ResourceManager::getInstance();
  const auto level = getCurrentLevel();

  int num_devices = rm.getNumDevices();
  if (num_devices < 2) {
    GTEST_SKIP() << "Test requires multiple GPUs";
  }

  if (level != umpire::IntrospectionLevel::Basic) {
    GTEST_SKIP() << "MultiGPU test specific to Basic mode";
  }

  // Test device 0
  {
    umpire::Allocator device0{rm.getAllocator("DEVICE::0")};
    void* ptr = device0.allocate(256);

    EXPECT_TRUE(rm.hasAllocator(ptr));
    auto retrieved = rm.getAllocator(ptr);
    // Should return DEVICE or DEVICE::0
    EXPECT_TRUE(retrieved.getName() == "DEVICE" ||
                retrieved.getName() == "DEVICE::0");

    device0.deallocate(ptr);
  }

  // Test device 1
  {
    umpire::Allocator device1{rm.getAllocator("DEVICE::1")};
    void* ptr = device1.allocate(256);

    EXPECT_TRUE(rm.hasAllocator(ptr));
    auto retrieved = rm.getAllocator(ptr);
    // Should return DEVICE::1 or fallback to DEVICE
    EXPECT_TRUE(retrieved.getName() == "DEVICE::1" ||
                retrieved.getName() == "DEVICE");

    device1.deallocate(ptr);
  }
}
#endif
