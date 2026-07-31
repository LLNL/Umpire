//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/quick_pool.hpp"
#include "umpire/resource/host_memory.hpp"
#include "umpire/memory.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

class test_memory : public umpire::memory {
public:
  using platform = umpire::host_platform;

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

  umpire::resource::Platform get_platform() const override
  {
    return umpire::resource::Platform::host;
  }
};

using pool_type = umpire::strategy::quick_pool<test_memory>;

} // namespace

// ============================================================================
// Construction
// ============================================================================

TEST(quick_pool, construct_with_defaults)
{
  test_memory parent;
  pool_type pool("qp_construct", &parent, 1024, 128, 16);

  EXPECT_EQ(pool.get_name(), "qp_construct");
  EXPECT_EQ(pool.get_actual_size(), 0);
  EXPECT_EQ(pool.get_total_blocks(), 0);
}

TEST(quick_pool, construct_with_nullptr_throws)
{
  EXPECT_THROW((pool_type("qp_null", nullptr, 1024, 128, 16)), std::invalid_argument);
}

TEST(quick_pool, invalid_heuristic_percentage_throws)
{
  EXPECT_THROW(pool_type::percent_releasable(-1), umpire::runtime_error);
  EXPECT_THROW(pool_type::percent_releasable(101), umpire::runtime_error);
  EXPECT_THROW(pool_type::percent_releasable_hwm(-1), umpire::runtime_error);
  EXPECT_THROW(pool_type::percent_releasable_hwm(101), umpire::runtime_error);
}

// ============================================================================
// Basic allocation behavior (ported from v1 primary_pool_tests expectations)
// ============================================================================

TEST(quick_pool, allocate_deallocate_updates_statistics)
{
  test_memory parent;
  pool_type pool("qp_stats", &parent, 1024, 128, 16);

  void* ptr = pool.allocate(64);
  ASSERT_NE(ptr, nullptr);

  // First block is first_minimum_pool_allocation_size bytes.
  EXPECT_EQ(pool.get_actual_size(), 1024);
  EXPECT_EQ(pool.get_current_size(), 64);
  EXPECT_EQ(pool.get_highwatermark(), 64);
  EXPECT_EQ(pool.get_total_blocks(), 1);
  EXPECT_EQ(pool.get_releasable_blocks(), 0);

  pool.deallocate(ptr);

  EXPECT_EQ(pool.get_current_size(), 0);
  EXPECT_EQ(pool.get_highwatermark(), 64);
  EXPECT_EQ(pool.get_releasable_blocks(), 1);
}

TEST(quick_pool, allocation_larger_than_block_size_grows_pool)
{
  test_memory parent;
  pool_type pool("qp_grow", &parent, 1024, 128, 16,
                 pool_type::percent_releasable(0));

  void* small = pool.allocate(64);
  void* large = pool.allocate(4096);

  EXPECT_EQ(pool.get_actual_size(), 1024 + 4096);
  EXPECT_EQ(pool.get_total_blocks(), 2);

  std::memset(large, 0xAB, 4096);
  EXPECT_EQ(static_cast<unsigned char*>(large)[4095], 0xAB);

  pool.deallocate(small);
  pool.deallocate(large);
}

TEST(quick_pool, best_fit_reuses_freed_chunk)
{
  test_memory parent;
  pool_type pool("qp_reuse", &parent, 1024, 128, 16,
                 pool_type::percent_releasable(0));

  void* a = pool.allocate(256);
  void* b = pool.allocate(256);
  pool.deallocate(a);

  // A same-size allocation must reuse the freed chunk (no pool growth).
  const std::size_t actual_before = pool.get_actual_size();
  void* c = pool.allocate(256);
  EXPECT_EQ(c, a);
  EXPECT_EQ(pool.get_actual_size(), actual_before);

  pool.deallocate(b);
  pool.deallocate(c);
}

TEST(quick_pool, unknown_pointer_deallocation_throws)
{
  test_memory parent;
  pool_type pool("qp_unknown", &parent, 1024, 128, 16);

  int on_stack{0};
  EXPECT_THROW(pool.deallocate(&on_stack), umpire::unknown_allocation);
}

TEST(quick_pool, largest_available_block)
{
  test_memory parent;
  pool_type pool("qp_largest", &parent, 1024, 128, 16,
                 pool_type::percent_releasable(0));

  EXPECT_EQ(pool.get_largest_available_block(), 0);

  void* ptr = pool.allocate(64);
  // 1024-byte block minus the 64-byte allocation leaves a 960-byte remainder.
  EXPECT_EQ(pool.get_largest_available_block(), 960);

  pool.deallocate(ptr);
  EXPECT_EQ(pool.get_largest_available_block(), 1024);
}

TEST(quick_pool, release_returns_free_blocks_to_parent)
{
  test_memory parent;
  pool_type pool("qp_release", &parent, 1024, 1024, 16,
                 pool_type::percent_releasable(0));

  std::vector<void*> ptrs;
  for (int i = 0; i < 4; ++i) {
    ptrs.push_back(pool.allocate(1024));
  }
  EXPECT_EQ(pool.get_total_blocks(), 4);
  EXPECT_EQ(parent.get_current_size(), pool.get_actual_size() + 4 * 16); // alignment padding

  for (void* p : ptrs) {
    pool.deallocate(p);
  }
  EXPECT_EQ(pool.get_releasable_blocks(), 4);

  pool.release();
  EXPECT_EQ(pool.get_total_blocks(), 0);
  EXPECT_EQ(pool.get_actual_size(), 0);
  EXPECT_EQ(parent.get_current_size(), 0);
}

// ============================================================================
// Heuristic behavior (ported from v1 pool_heuristics_tests.cpp)
// ============================================================================

TEST(quick_pool, percent_releasable_100)
{
  test_memory parent;
  pool_type pool("qp_pr100", &parent, 1024, 128, 16,
                 pool_type::percent_releasable(100));

  std::vector<void*> ptrs;
  const int max_blocks{9};

  for (int i{0}; i < max_blocks; i++) {
    ptrs.push_back(pool.allocate(1024));
    EXPECT_EQ(pool.get_releasable_blocks(), 0);
    EXPECT_EQ(pool.get_total_blocks(), i + 1);
  }

  for (int i{max_blocks - 1}; i > 0; i--) {
    pool.deallocate(ptrs[i]);
    EXPECT_EQ(pool.get_releasable_blocks(), max_blocks - i);
  }

  // Final deallocate makes 100% releasable: the pool coalesces to one block.
  pool.deallocate(ptrs[0]);
  EXPECT_EQ(pool.get_releasable_blocks(), 1);
  EXPECT_EQ(pool.get_total_blocks(), 1);
}

TEST(quick_pool, percent_releasable_hwm_25)
{
  test_memory parent;
  pool_type pool("qp_prhwm25", &parent, 1024, 128, 16,
                 pool_type::percent_releasable_hwm(25));

  std::vector<void*> ptrs;

  // allocate 64 bytes 23 times with first block 1024 bytes and next block 128 bytes
  for (int i{0}; i < 23; ++i) {
    ptrs.push_back(pool.allocate(64));
    EXPECT_EQ(pool.get_releasable_blocks(), 0);
  }

  EXPECT_EQ(pool.get_actual_size(), 1536);
  EXPECT_EQ(pool.get_highwatermark(), 1472);
  EXPECT_EQ(pool.get_total_blocks(), 5);

  // Deallocate 7*64 bytes so 25% of the pool is releasable; the pool
  // coalesces automatically to the aligned high watermark.
  for (int i{22}; i > 15; --i) {
    pool.deallocate(ptrs[i]);
  }

  EXPECT_EQ(pool.get_actual_size(), pool.get_highwatermark());
  EXPECT_EQ(pool.get_total_blocks(), 2);
  EXPECT_EQ(pool.get_releasable_blocks(), 1);

  pool.release();

  for (int i{16}; i > 0; --i) {
    pool.deallocate(ptrs[i - 1]);
  }

  EXPECT_EQ(pool.get_releasable_blocks(), 1);
  EXPECT_EQ(pool.get_total_blocks(), 1);
}

TEST(quick_pool, blocks_releasable_2)
{
  test_memory parent;
  pool_type pool("qp_br2", &parent, 1024, 128, 16,
                 pool_type::blocks_releasable(2));

  std::vector<void*> ptrs;
  const int max_blocks{9};

  for (int i{0}; i < max_blocks; i++) {
    ptrs.push_back(pool.allocate(1024));
    EXPECT_EQ(pool.get_releasable_blocks(), 0);
    EXPECT_EQ(pool.get_total_blocks(), i + 1);
  }

  // Each deallocation makes a whole block releasable; on reaching two the
  // pool coalesces back down to a single free block.
  for (int i{max_blocks}; i > 0; i--) {
    pool.deallocate(ptrs[i - 1]);
    EXPECT_EQ(pool.get_releasable_blocks(), 1);
  }

  EXPECT_EQ(pool.get_releasable_blocks(), 1);
  EXPECT_EQ(pool.get_total_blocks(), 1);
}

TEST(quick_pool, blocks_releasable_hwm_2)
{
  test_memory parent;
  pool_type pool("qp_brhwm2", &parent, 1024, 128, 16,
                 pool_type::blocks_releasable_hwm(2));

  std::vector<void*> ptrs;

  for (int i{0}; i < 23; ++i) {
    ptrs.push_back(pool.allocate(64));
    EXPECT_EQ(pool.get_releasable_blocks(), 0);
  }

  EXPECT_EQ(pool.get_actual_size(), 1536);
  EXPECT_EQ(pool.get_highwatermark(), 1472);
  EXPECT_EQ(pool.get_total_blocks(), 5);

  // Deallocate 4 times so two blocks are releasable; the pool coalesces
  // automatically.
  for (int i{22}; i > 18; --i) {
    pool.deallocate(ptrs[i]);
  }

  EXPECT_EQ(pool.get_actual_size(), pool.get_highwatermark());
  EXPECT_EQ(pool.get_total_blocks(), 4);
  EXPECT_EQ(pool.get_releasable_blocks(), 1);

  for (int i{19}; i > 0; --i) {
    pool.deallocate(ptrs[i - 1]);
  }

  EXPECT_EQ(pool.get_releasable_blocks(), 1);
  EXPECT_EQ(pool.get_total_blocks(), 1);
}

TEST(quick_pool, explicit_coalesce)
{
  test_memory parent;
  pool_type pool("qp_coalesce", &parent, 1024, 1024, 16,
                 pool_type::percent_releasable(0));

  std::vector<void*> ptrs;
  for (int i = 0; i < 4; ++i) {
    ptrs.push_back(pool.allocate(1024));
  }
  for (void* p : ptrs) {
    pool.deallocate(p);
  }

  // percent_releasable(0) never fires automatically.
  EXPECT_EQ(pool.get_total_blocks(), 4);

  // With the zero heuristic explicit coalesce() is also a no-op.
  pool.coalesce();
  EXPECT_EQ(pool.get_total_blocks(), 4);

  pool.release();
  EXPECT_EQ(pool.get_total_blocks(), 0);
}

// ============================================================================
// Alignment
// ============================================================================

TEST(quick_pool, alignment_is_respected)
{
  test_memory parent;
  pool_type pool("qp_align", &parent, 4096, 4096, 256,
                 pool_type::percent_releasable(0));

  std::vector<void*> ptrs;
  for (int i = 0; i < 8; ++i) {
    void* p = pool.allocate(100);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(p) % 256, 0u);
    ptrs.push_back(p);
  }
  for (void* p : ptrs) {
    pool.deallocate(p);
  }
}

TEST(quick_pool, aligned_size_tracks_rounded_bytes)
{
  test_memory parent;
  pool_type pool("qp_aligned_size", &parent, 4096, 4096, 256,
                 pool_type::percent_releasable(0));

  void* p = pool.allocate(100);
  EXPECT_EQ(pool.get_aligned_size(), 256);
  EXPECT_EQ(pool.get_aligned_highwatermark(), 256);
  EXPECT_EQ(pool.get_current_size(), 100);

  pool.deallocate(p);
  EXPECT_EQ(pool.get_aligned_size(), 0);
  EXPECT_EQ(pool.get_aligned_highwatermark(), 256);
}

// ============================================================================
// Platform propagation
// ============================================================================

TEST(quick_pool, platform_propagation)
{
  static_assert(std::is_same_v<pool_type::platform, umpire::host_platform>,
                "quick_pool must propagate the platform of the wrapped memory");

  test_memory parent;
  pool_type pool("qp_platform", &parent, 1024, 128, 16);
  EXPECT_EQ(pool.get_platform(), umpire::resource::Platform::host);
}

// ============================================================================
// Data integrity
// ============================================================================

TEST(quick_pool, allocated_memory_is_writable_across_growth_and_coalesce)
{
  test_memory parent;
  pool_type pool("qp_integrity", &parent, 1024, 128, 16,
                 pool_type::blocks_releasable(2));

  std::vector<void*> ptrs;
  for (int i = 0; i < 16; ++i) {
    void* p = pool.allocate(96);
    std::memset(p, i, 96);
    ptrs.push_back(p);
  }

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(static_cast<unsigned char*>(ptrs[i])[0], i);
    EXPECT_EQ(static_cast<unsigned char*>(ptrs[i])[95], i);
  }

  for (int i = 0; i < 16; i += 2) {
    pool.deallocate(ptrs[i]);
  }
  // Odd-indexed allocations survive interleaved frees + heuristic coalesce.
  for (int i = 1; i < 16; i += 2) {
    EXPECT_EQ(static_cast<unsigned char*>(ptrs[i])[0], i);
    EXPECT_EQ(static_cast<unsigned char*>(ptrs[i])[95], i);
  }
  for (int i = 1; i < 16; i += 2) {
    pool.deallocate(ptrs[i]);
  }
}
