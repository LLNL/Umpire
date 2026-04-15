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

TEST(IntrospectionLevelTest, OnTracksNamedAllocationMetadata)
{
  auto& rm = umpire::ResourceManager::getInstance();
  IntrospectionLevelGuard guard{rm};

  umpire::Allocator allocator{rm.getAllocator("HOST")};

  const std::string alloc_name{"my_named_alloc"};
  constexpr std::size_t size{64};

  rm.setIntrospectionLevel(umpire::IntrospectionLevel::On);
  {
    void* p = allocator.allocate(alloc_name, size);
    ASSERT_TRUE(rm.hasAllocator(p));
    EXPECT_NO_THROW(rm.getAllocator(p));
    EXPECT_EQ(rm.getSize(p), size);
    EXPECT_EQ(rm.findAllocationRecord(p)->name, alloc_name);
    allocator.deallocate(p);
  }
}

TEST(IntrospectionLevelTest, BasicTracksExactPointerOwnershipOnly)
{
  auto& rm = umpire::ResourceManager::getInstance();
  IntrospectionLevelGuard guard{rm};

  umpire::Allocator allocator{rm.getAllocator("HOST")};

  const std::string alloc_name{"my_named_alloc"};
  constexpr std::size_t size{64};

  rm.setIntrospectionLevel(umpire::IntrospectionLevel::Basic);

  {
    void* p = allocator.allocate(alloc_name, size);
    ASSERT_TRUE(rm.hasAllocator(p));
    EXPECT_FALSE(rm.hasAllocator(static_cast<char*>(p) + 1));
    EXPECT_THROW(rm.findAllocationRecord(p), umpire::runtime_error);
    EXPECT_NO_THROW(rm.getAllocator(p));  // Works via API inference
    EXPECT_THROW(rm.getSize(p), umpire::runtime_error);
    EXPECT_THROW(umpire::get_allocator_records(allocator), umpire::runtime_error);
    allocator.deallocate(p);
  }
}

TEST(IntrospectionLevelTest, OffDisablesPublicOwnershipQueries)
{
  auto& rm = umpire::ResourceManager::getInstance();
  IntrospectionLevelGuard guard{rm};

  umpire::Allocator allocator{rm.getAllocator("HOST")};
  constexpr std::size_t size{64};

  rm.setIntrospectionLevel(umpire::IntrospectionLevel::Off);

  {
    void* p = allocator.allocate(size);
    EXPECT_FALSE(rm.hasAllocator(p));
    EXPECT_THROW(rm.findAllocationRecord(p), umpire::runtime_error);
    EXPECT_THROW(rm.getAllocator(p), umpire::runtime_error);
    EXPECT_THROW(rm.getSize(p), umpire::runtime_error);
    allocator.deallocate(p);
  }
}
{
  auto& rm = umpire::ResourceManager::getInstance();
  IntrospectionLevelGuard guard{rm};

  int stack_var = 42;
  void* stack_ptr = &stack_var;

  // Off mode
  rm.setIntrospectionLevel(umpire::IntrospectionLevel::Off);
  EXPECT_FALSE(rm.hasAllocator(stack_ptr));

  // Basic mode - runtime API will likely fail, fallback to HOST
  rm.setIntrospectionLevel(umpire::IntrospectionLevel::Basic);
  // May return true (infers HOST) or false (API fails)
  // Either is acceptable for non-Umpire pointer
  bool has_alloc = rm.hasAllocator(stack_ptr);
  // Should not crash
  (void)has_alloc; // Suppress unused warning

  // On mode
  rm.setIntrospectionLevel(umpire::IntrospectionLevel::On);
  EXPECT_FALSE(rm.hasAllocator(stack_ptr));
  EXPECT_THROW(rm.getAllocator(stack_ptr), umpire::runtime_error);
}

TEST(IntrospectionLevelTest, EdgeCaseFreedPointer)
{
  auto& rm = umpire::ResourceManager::getInstance();
  IntrospectionLevelGuard guard{rm};

  umpire::Allocator allocator{rm.getAllocator("HOST")};

  // Test On mode
  rm.setIntrospectionLevel(umpire::IntrospectionLevel::On);
  {
    void* ptr = allocator.allocate(256);
    EXPECT_TRUE(rm.hasAllocator(ptr));

    allocator.deallocate(ptr);

    // After deallocation, should not be tracked
    EXPECT_FALSE(rm.hasAllocator(ptr));
    EXPECT_THROW(rm.getAllocator(ptr), umpire::runtime_error);
  }

  // Test Basic mode
  rm.setIntrospectionLevel(umpire::IntrospectionLevel::Basic);
  {
    void* ptr = allocator.allocate(256);
    // Note: In Basic mode, we don't track, so behavior after free is undefined
    // but shouldn't crash
    allocator.deallocate(ptr);

    // Querying freed pointer may or may not work (depends on OS reuse)
    // Just verify it doesn't crash
    bool has_alloc = rm.hasAllocator(ptr);
    (void)has_alloc; // Suppress unused warning
  }
}

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
TEST(IntrospectionLevelTest, BasicModeAsyncCopy)
{
  auto& rm = umpire::ResourceManager::getInstance();
  IntrospectionLevelGuard guard{rm};

  rm.setIntrospectionLevel(umpire::IntrospectionLevel::Basic);

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
TEST(IntrospectionLevelTest, BasicModeAsyncMemset)
{
  auto& rm = umpire::ResourceManager::getInstance();
  IntrospectionLevelGuard guard{rm};

  rm.setIntrospectionLevel(umpire::IntrospectionLevel::Basic);

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

TEST(IntrospectionLevelTest, BasicModePoolAllocatorIdentification)
{
  auto& rm = umpire::ResourceManager::getInstance();
  IntrospectionLevelGuard guard{rm};

  umpire::Allocator host_alloc{rm.getAllocator("HOST")};

  // Create a pool backed by HOST
  auto pool = rm.makeAllocator<umpire::strategy::QuickPool>("TestPool", host_alloc);

  void* pool_ptr = pool.allocate(256);

  // Test On mode - should return the specific pool
  rm.setIntrospectionLevel(umpire::IntrospectionLevel::On);
  {
    auto retrieved = rm.getAllocator(pool_ptr);
    EXPECT_EQ(retrieved.getName(), "TestPool");
  }

  // Test Basic mode - should return HOST (the backing resource)
  rm.setIntrospectionLevel(umpire::IntrospectionLevel::Basic);
  {
    auto retrieved = rm.getAllocator(pool_ptr);
    // Basic mode returns backing resource, not the pool
    EXPECT_EQ(retrieved.getName(), "HOST");
  }

  pool.deallocate(pool_ptr);
}

TEST(IntrospectionLevelTest, ErrorMessageQuality)
{
  auto& rm = umpire::ResourceManager::getInstance();
  IntrospectionLevelGuard guard{rm};

  umpire::Allocator allocator{rm.getAllocator("HOST")};

  // Basic mode size=0 error message
  rm.setIntrospectionLevel(umpire::IntrospectionLevel::Basic);
  {
    void* p1 = allocator.allocate(256);
    void* p2 = allocator.allocate(256);

    try {
      rm.copy(p2, p1, 0);
      FAIL() << "Expected runtime_error";
    } catch (const umpire::runtime_error& e) {
      std::string msg = e.what();
      EXPECT_TRUE(msg.find("size=0") != std::string::npos ||
                  msg.find("auto-sizing") != std::string::npos)
        << "Error message should mention size=0 or auto-sizing: " << msg;
    }

    allocator.deallocate(p1);
    allocator.deallocate(p2);
  }

  // Off mode operation error message
  rm.setIntrospectionLevel(umpire::IntrospectionLevel::Off);
  {
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
  }
}

#if defined(UMPIRE_ENABLE_CUDA) || defined(UMPIRE_ENABLE_HIP)
TEST(IntrospectionLevelTest, BasicModeMultiGPU)
{
  auto& rm = umpire::ResourceManager::getInstance();
  IntrospectionLevelGuard guard{rm};

  int num_devices = rm.getNumDevices();
  if (num_devices < 2) {
    GTEST_SKIP() << "Test requires multiple GPUs";
  }

  rm.setIntrospectionLevel(umpire::IntrospectionLevel::Basic);

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
>>>>>>> ea8e4485 (Use pointer-only approach for basic introspection)
