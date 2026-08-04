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
#include "umpire/config.hpp"
#include "umpire/detail/registry.hpp"
#include "umpire/error.hpp"
#include "umpire/resource/host_memory.hpp"
#include "umpire/resource/v2_backed_resource.hpp"
#include "umpire/strategy/SizeLimiter.hpp"
#include "umpire/strategy/ThreadSafeAllocator.hpp"

#include "gtest/gtest.h"

#include <atomic>
#include <cstddef>
#include <cstring>
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

//////////////////////////////////////////////////////////////////////////////
// v1 RESOURCE -> API v2 delegation tests (resource::v2_backed_resource)
//
// The tests below exercise v1's *resources* (as opposed to the *strategies*
// tested above) being backed by API v2 memory objects via
// umpire::resource::v2_backed_resource -- see
// src/umpire/resource/v2_backed_resource.hpp for the adapter itself, and the
// per-resource factories (e.g. src/umpire/resource/HostResourceFactory.cpp)
// for the wiring under UMPIRE_V1_DELEGATE_TO_V2.
//////////////////////////////////////////////////////////////////////////////

// rm.getAllocator("HOST") is backed by a v2_backed_resource wrapping a
// Tracking=false umpire::resource::fast_host_memory instance (unless
// UMPIRE_ENABLE_NUMA is set, in which case HOST stays NATIVE -- see
// HostResourceFactory.cpp). Basic allocate/deallocate and introspection
// (getCurrentSize/getHighWatermark/getAllocationCount) must behave exactly
// as the native path, since v1's counters live in the outer
// AllocationStrategy and are untouched by delegation.
TEST(V1Delegation, DelegatedHostResourceAllocateDeallocateAndIntrospection)
{
#if !defined(UMPIRE_ENABLE_NUMA)
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");

  const std::size_t initial_current = host_allocator.getCurrentSize();
  const std::size_t initial_count = host_allocator.getAllocationCount();

  void* ptr = host_allocator.allocate(1024);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(host_allocator.getCurrentSize(), initial_current + 1024);
  EXPECT_EQ(host_allocator.getAllocationCount(), initial_count + 1);
  EXPECT_GE(host_allocator.getHighWatermark(), initial_current + 1024);
  EXPECT_TRUE(rm.hasAllocator(ptr));

  auto* record = rm.findAllocationRecord(ptr);
  ASSERT_NE(record, nullptr);
  EXPECT_EQ(record->ptr, ptr);
  EXPECT_EQ(record->size, 1024u);

  host_allocator.deallocate(ptr);

  EXPECT_EQ(host_allocator.getCurrentSize(), initial_current);
  EXPECT_EQ(host_allocator.getAllocationCount(), initial_count);
  EXPECT_FALSE(rm.hasAllocator(ptr));
#else
  GTEST_SKIP() << "HOST stays NATIVE under UMPIRE_ENABLE_NUMA; nothing to delegate.";
#endif
}

// A delegated HOST allocation must be a fully usable buffer through v1's
// ResourceManager::memset()/copy() operations -- these dispatch on
// getAllocationStrategy()->getPlatform(), and v2_backed_resource::getPlatform()
// returns the platform supplied by the factory (Platform::host here), so the
// native host memset/copy operations are selected exactly as for the
// non-delegated path.
TEST(V1Delegation, DelegatedHostResourceSupportsMemsetAndCopyThroughResourceManager)
{
#if !defined(UMPIRE_ENABLE_NUMA)
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");

  constexpr std::size_t kSize = 256;
  void* src = host_allocator.allocate(kSize);
  void* dst = host_allocator.allocate(kSize);
  ASSERT_NE(src, nullptr);
  ASSERT_NE(dst, nullptr);

  rm.memset(src, 0x5A, kSize);

  auto* src_bytes = static_cast<unsigned char*>(src);
  for (std::size_t i = 0; i < kSize; ++i) {
    ASSERT_EQ(src_bytes[i], 0x5A);
  }

  std::memset(dst, 0, kSize);
  rm.copy(dst, src, kSize);

  auto* dst_bytes = static_cast<unsigned char*>(dst);
  for (std::size_t i = 0; i < kSize; ++i) {
    ASSERT_EQ(dst_bytes[i], 0x5A);
  }

  host_allocator.deallocate(src);
  host_allocator.deallocate(dst);
#else
  GTEST_SKIP() << "HOST stays NATIVE under UMPIRE_ENABLE_NUMA; nothing to delegate.";
#endif
}

// The v2_backed_resource wraps a NAMED (non-singleton) v2 memory instance,
// distinct from umpire::resource::host_memory::get() (name "HOST") and from
// the v1-visible resource name itself. Confirm that identity directly via
// v2_memory(), and confirm the wrapped instance's name carries the
// "_v2backed" suffix documented in v2_backed_resource.hpp (this is what
// prevents the HOST-only v1<->v2 interop bridge in src/umpire/memory.cpp,
// which keys off get_name() == "HOST", from firing for delegated
// allocations).
TEST(V1Delegation, DelegatedHostResourceWrapsDistinctlyNamedV2Instance)
{
#if !defined(UMPIRE_ENABLE_NUMA)
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");

  auto* strategy = host_allocator.getAllocationStrategy();
  ASSERT_NE(strategy, nullptr);

  auto* v2_backed = dynamic_cast<umpire::resource::v2_backed_resource*>(strategy);
  ASSERT_NE(v2_backed, nullptr) << "Expected HOST to be backed by v2_backed_resource under "
                                   "UMPIRE_V1_DELEGATE_TO_V2 (NUMA disabled).";

  ASSERT_NE(v2_backed->v2_memory(), nullptr);
  EXPECT_NE(v2_backed->v2_memory()->get_name(), "HOST");
  EXPECT_NE(static_cast<umpire::memory*>(v2_backed->v2_memory()),
            static_cast<umpire::memory*>(&umpire::resource::host_memory<>::get()));
#else
  GTEST_SKIP() << "HOST stays NATIVE under UMPIRE_ENABLE_NUMA; nothing to delegate.";
#endif
}

#if defined(UMPIRE_ENABLE_FILE_RESOURCE) && !defined(UMPIRE_ENABLE_UMAP)
// FILE resources are delegated to v2's file_memory unless UMAP support is
// enabled (v2's file_memory has no UMAP-backed path -- see
// FileMemoryResourceFactory.cpp). Verify a full mmap-backed allocate/write/
// read/deallocate round trip through the delegated resource.
TEST(V1Delegation, DelegatedFileResourceAllocateWriteReadDeallocateRoundTrip)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto file_allocator = rm.getAllocator("FILE");

  constexpr std::size_t kSize = 4096;
  void* ptr = file_allocator.allocate(kSize);
  ASSERT_NE(ptr, nullptr);

  auto* bytes = static_cast<unsigned char*>(ptr);
  for (std::size_t i = 0; i < kSize; ++i) {
    bytes[i] = static_cast<unsigned char>(i % 256);
  }
  for (std::size_t i = 0; i < kSize; ++i) {
    ASSERT_EQ(bytes[i], static_cast<unsigned char>(i % 256));
  }

  EXPECT_TRUE(rm.hasAllocator(ptr));
  file_allocator.deallocate(ptr);
  EXPECT_FALSE(rm.hasAllocator(ptr));
}
#endif // UMPIRE_ENABLE_FILE_RESOURCE && !UMPIRE_ENABLE_UMAP
