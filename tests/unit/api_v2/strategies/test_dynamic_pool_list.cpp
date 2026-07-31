//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/strategy/dynamic_pool_list.hpp"
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

using pool_type = umpire::strategy::dynamic_pool_list<test_memory>;

} // namespace

// ============================================================================
// Construction
// ============================================================================

TEST(dynamic_pool_list, construct_with_defaults)
{
  test_memory parent;
  pool_type pool("dpl_construct", &parent, 1024, 128, 16);

  EXPECT_EQ(pool.get_name(), "dpl_construct");
  EXPECT_EQ(pool.get_actual_size(), 0);
  EXPECT_EQ(pool.get_total_blocks(), 0);
}

TEST(dynamic_pool_list, construct_with_nullptr_throws)
{
  EXPECT_THROW((pool_type("dpl_null", nullptr, 1024, 128, 16)), std::invalid_argument);
}

TEST(dynamic_pool_list, invalid_heuristic_percentage_throws)
{
  EXPECT_THROW(pool_type::percent_releasable(-1), umpire::runtime_error);
  EXPECT_THROW(pool_type::percent_releasable(101), umpire::runtime_error);
  EXPECT_THROW(pool_type::percent_releasable_hwm(-1), umpire::runtime_error);
  EXPECT_THROW(pool_type::percent_releasable_hwm(101), umpire::runtime_error);
}

// ============================================================================
// Basic allocation behavior (ported from v1 primary_pool_tests expectations)
// ============================================================================

TEST(dynamic_pool_list, allocate_deallocate_updates_statistics)
{
  test_memory parent;
  pool_type pool("dpl_stats", &parent, 1024, 128, 16);

  void* ptr = pool.allocate(64);
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(pool.get_actual_size(), 1024);
  EXPECT_EQ(pool.get_current_size(), 64);
  EXPECT_EQ(pool.get_highwatermark(), 64);
  EXPECT_EQ(pool.get_total_blocks(), 1);
  EXPECT_EQ(pool.get_releasable_blocks(), 0);

  pool.deallocate(ptr);

  EXPECT_EQ(pool.get_current_size(), 0);
  EXPECT_EQ(pool.get_highwatermark(), 64);
}

TEST(dynamic_pool_list, best_fit_reuses_freed_block)
{
  test_memory parent;
  pool_type pool("dpl_reuse", &parent, 1024, 128, 16,
                 pool_type::percent_releasable(0));

  void* a = pool.allocate(256);
  void* b = pool.allocate(256);
  pool.deallocate(a);

  const std::size_t actual_before = pool.get_actual_size();
  void* c = pool.allocate(256);
  EXPECT_EQ(c, a);
  EXPECT_EQ(pool.get_actual_size(), actual_before);

  pool.deallocate(b);
  pool.deallocate(c);
}

TEST(dynamic_pool_list, unknown_pointer_deallocation_throws)
{
  test_memory parent;
  pool_type pool("dpl_unknown", &parent, 1024, 128, 16);

  int on_stack{0};
  EXPECT_THROW(pool.deallocate(&on_stack), umpire::unknown_allocation);
}

TEST(dynamic_pool_list, largest_available_block)
{
  test_memory parent;
  pool_type pool("dpl_largest", &parent, 1024, 128, 16,
                 pool_type::percent_releasable(0));

  EXPECT_EQ(pool.get_largest_available_block(), 0);

  void* ptr = pool.allocate(64);
  EXPECT_EQ(pool.get_largest_available_block(), 960);

  pool.deallocate(ptr);
  EXPECT_EQ(pool.get_largest_available_block(), 1024);
}

TEST(dynamic_pool_list, release_returns_free_blocks_to_parent)
{
  test_memory parent;
  pool_type pool("dpl_release", &parent, 1024, 1024, 16,
                 pool_type::percent_releasable(0));

  std::vector<void*> ptrs;
  for (int i = 0; i < 4; ++i) {
    ptrs.push_back(pool.allocate(1024));
  }
  EXPECT_EQ(pool.get_total_blocks(), 4);

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

TEST(dynamic_pool_list, percent_releasable_100)
{
  test_memory parent;
  pool_type pool("dpl_pr100", &parent, 1024, 128, 16,
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

  // Final deallocate empties the pool: it coalesces down to one block.
  pool.deallocate(ptrs[0]);
  EXPECT_EQ(pool.get_releasable_blocks(), 1);
  EXPECT_EQ(pool.get_total_blocks(), 1);
}

TEST(dynamic_pool_list, percent_releasable_hwm_25)
{
  test_memory parent;
  pool_type pool("dpl_prhwm25", &parent, 1024, 128, 16,
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

TEST(dynamic_pool_list, blocks_releasable_2)
{
  test_memory parent;
  pool_type pool("dpl_br2", &parent, 1024, 128, 16,
                 pool_type::blocks_releasable(2));

  std::vector<void*> ptrs;
  const int max_blocks{9};

  for (int i{0}; i < max_blocks; i++) {
    ptrs.push_back(pool.allocate(1024));
    EXPECT_EQ(pool.get_releasable_blocks(), 0);
    EXPECT_EQ(pool.get_total_blocks(), i + 1);
  }

  for (int i{max_blocks}; i > 0; i--) {
    pool.deallocate(ptrs[i - 1]);
    EXPECT_EQ(pool.get_releasable_blocks(), 1);
  }

  EXPECT_EQ(pool.get_releasable_blocks(), 1);
  EXPECT_EQ(pool.get_total_blocks(), 1);
}

TEST(dynamic_pool_list, blocks_releasable_hwm_2)
{
  test_memory parent;
  pool_type pool("dpl_brhwm2", &parent, 1024, 128, 16,
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

// ============================================================================
// Alignment
// ============================================================================

TEST(dynamic_pool_list, alignment_is_respected)
{
  test_memory parent;
  pool_type pool("dpl_align", &parent, 4096, 4096, 256,
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

TEST(dynamic_pool_list, aligned_size_tracks_rounded_bytes)
{
  test_memory parent;
  pool_type pool("dpl_aligned_size", &parent, 4096, 4096, 256,
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

TEST(dynamic_pool_list, platform_propagation)
{
  static_assert(std::is_same_v<pool_type::platform, umpire::host_platform>,
                "dynamic_pool_list must propagate the platform of the wrapped memory");

  test_memory parent;
  pool_type pool("dpl_platform", &parent, 1024, 128, 16);
  EXPECT_EQ(pool.get_platform(), umpire::resource::Platform::host);
}

// ============================================================================
// Data integrity
// ============================================================================

TEST(dynamic_pool_list, allocated_memory_is_writable_across_growth_and_coalesce)
{
  test_memory parent;
  pool_type pool("dpl_integrity", &parent, 1024, 128, 16,
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
