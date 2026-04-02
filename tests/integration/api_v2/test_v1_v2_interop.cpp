//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/Umpire.hpp"
#include "umpire/detail/registry.hpp"
#include "umpire/resource/host_memory.hpp"

#include "camp/resource/host.hpp"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace {

using host_memory = umpire::resource::host_memory<>;

host_memory& host()
{
  return host_memory::get();
}

} // namespace

TEST(ApiV1V2Interop, V2HostAllocationsAppearInV1ResourceManager)
{
  auto& rm = umpire::ResourceManager::getInstance();
  void* ptr = host().allocate(64);

  ASSERT_TRUE(rm.hasAllocator(ptr));

  auto record = rm.findAllocationRecord(ptr);
  ASSERT_NE(record, nullptr);
  EXPECT_EQ(record->ptr, ptr);
  EXPECT_EQ(record->size, 64u);
  EXPECT_EQ(rm.getAllocator(ptr).getName(), "HOST");

  host().deallocate(ptr);
  EXPECT_FALSE(rm.hasAllocator(ptr));
}

TEST(ApiV1V2Interop, V1MemsetOperatesOnV2HostAllocation)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto* bytes = static_cast<unsigned char*>(host().allocate(16));

  std::fill(bytes, bytes + 16, static_cast<unsigned char>(0xAB));
  rm.memset(bytes, 0x11, 16);

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(bytes[i], 0x11);
  }

  host().deallocate(bytes);
}

TEST(ApiV1V2Interop, V1CopyOperatesAcrossV1AndV2HostAllocations)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");

  auto* src = static_cast<unsigned char*>(host().allocate(16));
  auto* dst = static_cast<unsigned char*>(host_allocator.allocate(16));

  for (int i = 0; i < 16; ++i) {
    src[i] = static_cast<unsigned char>(i + 3);
    dst[i] = 0;
  }

  rm.copy(dst, src, 16);

  std::array<unsigned char, 16> expected{};
  for (int i = 0; i < 16; ++i) {
    expected[static_cast<std::size_t>(i)] = static_cast<unsigned char>(i + 3);
  }

  EXPECT_TRUE(std::equal(dst, dst + 16, expected.begin()));

  host().deallocate(src);
  host_allocator.deallocate(dst);
}

TEST(ApiV1V2Interop, V1DeallocateUsesV2OwnerForHostAllocation)
{
  auto& rm = umpire::ResourceManager::getInstance();
  void* ptr = host().allocate(64);

  ASSERT_TRUE(rm.hasAllocator(ptr));
  ASSERT_TRUE(umpire::detail::registry::get().find_allocation(ptr).has_value());
  EXPECT_EQ(host().get_current_size(), 64u);

  rm.deallocate(ptr);

  EXPECT_FALSE(rm.hasAllocator(ptr));
  EXPECT_FALSE(umpire::detail::registry::get().find_allocation(ptr).has_value());
  EXPECT_EQ(host().get_current_size(), 0u);
}

TEST(ApiV1V2Interop, V1ZeroSizeReallocateReleasesV2HostAllocation)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto* ptr = static_cast<unsigned char*>(host().allocate(32));
  ASSERT_NE(ptr, nullptr);

  std::fill(ptr, ptr + 32, static_cast<unsigned char>(0x5A));
  void* zero = rm.reallocate(ptr, 0);

  EXPECT_EQ(host().get_current_size(), 0u);
  EXPECT_FALSE(umpire::detail::registry::get().find_allocation(ptr).has_value());

  if (zero) {
    rm.deallocate(zero);
  }
}

TEST(ApiV1V2Interop, V1ZeroSizeAsyncReallocateReleasesV2HostAllocation)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto* ptr = static_cast<unsigned char*>(host().allocate(24));
  ASSERT_NE(ptr, nullptr);

  camp::resources::Resource ctx{camp::resources::Host{}};
  void* zero = rm.reallocate(ptr, 0, ctx);

  EXPECT_EQ(host().get_current_size(), 0u);
  EXPECT_FALSE(umpire::detail::registry::get().find_allocation(ptr).has_value());

  if (zero) {
    rm.deallocate(zero);
  }
}
