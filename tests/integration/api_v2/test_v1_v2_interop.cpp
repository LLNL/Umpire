//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/Umpire.hpp"
#include "umpire/detail/registry.hpp"
#include "umpire/resource/host_memory.hpp"
#include "umpire/strategy/NamedAllocationStrategy.hpp"

#include "camp/resource/host.hpp"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace {

using host_memory = umpire::resource::host_memory<>;

host_memory& host()
{
  return host_memory::get();
}

std::string unique_allocator_name(const char* prefix)
{
  static int counter = 0;
  return std::string{prefix} + "_" + std::to_string(counter++);
}

bool has_allocator_at(std::uintptr_t address)
{
  // NOLINTNEXTLINE(clang-analyzer-unix.Malloc)
  return umpire::ResourceManager::getInstance().hasAllocator(reinterpret_cast<void*>(address));
}

bool contains_record(const std::vector<umpire::util::AllocationRecord>& records, void* ptr, std::size_t size)
{
  return std::any_of(records.begin(), records.end(), [ptr, size](const auto& record) {
    return record.ptr == ptr && record.size == size;
  });
}

} // namespace

TEST(ApiV1V2Interop, V2HostAllocationsAppearInV1ResourceManager)
{
  auto& rm = umpire::ResourceManager::getInstance();
  void* ptr = host().allocate(64);
  auto ptr_value = reinterpret_cast<std::uintptr_t>(ptr);

  ASSERT_TRUE(rm.hasAllocator(ptr));

  auto record = rm.findAllocationRecord(ptr);
  ASSERT_NE(record, nullptr);
  EXPECT_EQ(record->ptr, ptr);
  EXPECT_EQ(record->size, 64u);
  EXPECT_EQ(rm.getAllocator(ptr).getName(), "HOST");

  host().deallocate(ptr);
  EXPECT_FALSE(has_allocator_at(ptr_value));
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

TEST(ApiV1V2Interop, V1IntrospectionReportsBridgedHostAllocationLifecycle)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");
  auto* ptr = static_cast<unsigned char*>(host().allocate(48));
  auto* offset_ptr = ptr + 7;
  auto ptr_value = reinterpret_cast<std::uintptr_t>(ptr);
  auto offset_ptr_value = reinterpret_cast<std::uintptr_t>(offset_ptr);

  EXPECT_TRUE(rm.hasAllocator(ptr));
  EXPECT_TRUE(rm.hasAllocator(offset_ptr));
  auto* base_record = rm.findAllocationRecord(ptr);
  auto* offset_record = rm.findAllocationRecord(offset_ptr);

  ASSERT_NE(base_record, nullptr);
  ASSERT_NE(offset_record, nullptr);
  EXPECT_EQ(base_record->ptr, ptr);
  EXPECT_EQ(offset_record->ptr, ptr);
  EXPECT_EQ(base_record->size, 48u);
  EXPECT_EQ(offset_record->size, 48u);
  EXPECT_EQ(rm.getAllocator(ptr).getId(), host_allocator.getId());
  EXPECT_EQ(rm.getAllocator(offset_ptr).getId(), host_allocator.getId());

  auto allocator_records = umpire::get_allocator_records(host_allocator);
  auto leaked_records = umpire::get_leaked_allocations(host_allocator);

  EXPECT_TRUE(contains_record(allocator_records, ptr, 48u));
  EXPECT_TRUE(contains_record(leaked_records, ptr, 48u));

  std::ostringstream report;
  umpire::print_allocator_records(host_allocator, report);

  std::ostringstream ptr_text;
  ptr_text << static_cast<void*>(ptr);
  EXPECT_NE(report.str().find("Allocations for HOST allocator:"), std::string::npos);
  EXPECT_NE(report.str().find(ptr_text.str()), std::string::npos);

  host().deallocate(ptr);

  EXPECT_FALSE(has_allocator_at(ptr_value));
  EXPECT_FALSE(has_allocator_at(offset_ptr_value));
  EXPECT_THROW(rm.findAllocationRecord(reinterpret_cast<void*>(ptr_value)), umpire::unknown_allocation);
  EXPECT_THROW(rm.findAllocationRecord(reinterpret_cast<void*>(offset_ptr_value)), umpire::unknown_allocation);
  EXPECT_THROW(rm.getAllocator(reinterpret_cast<void*>(ptr_value)), umpire::unknown_allocation);
  EXPECT_THROW(rm.getAllocator(reinterpret_cast<void*>(offset_ptr_value)), umpire::unknown_allocation);

  allocator_records = umpire::get_allocator_records(host_allocator);
  leaked_records = umpire::get_leaked_allocations(host_allocator);
  EXPECT_FALSE(contains_record(allocator_records, ptr, 48u));
  EXPECT_FALSE(contains_record(leaked_records, ptr, 48u));

  std::ostringstream cleared_report;
  umpire::print_allocator_records(host_allocator, cleared_report);
  EXPECT_TRUE(cleared_report.str().empty());
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

TEST(ApiV1V2Interop, V1MoveToHostShortCircuitsAndPreservesV2Ownership)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");
  auto* ptr = static_cast<unsigned char*>(host().allocate(24));

  ASSERT_TRUE(umpire::detail::registry::get().find_allocation(ptr).has_value());

  void* moved = rm.move(ptr, host_allocator);

  EXPECT_EQ(moved, ptr);
  EXPECT_TRUE(umpire::detail::registry::get().find_allocation(moved).has_value());
  EXPECT_EQ(host().get_current_size(), 24u);

  rm.deallocate(moved);
}

TEST(ApiV1V2Interop, V1MoveToDistinctAllocatorTransfersOwnershipToV1)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");
  auto named_allocator = rm.makeAllocator<umpire::strategy::NamedAllocationStrategy>(
      unique_allocator_name("API_V2_MOVED_HOST"), host_allocator);

  auto* ptr = static_cast<unsigned char*>(host().allocate(24));
  for (int i = 0; i < 24; ++i) {
    ptr[i] = static_cast<unsigned char>(i + 11);
  }

  void* moved = rm.move(ptr, named_allocator);
  auto* moved_bytes = static_cast<unsigned char*>(moved);

  ASSERT_NE(moved, nullptr);
  EXPECT_NE(moved, ptr);
  EXPECT_FALSE(umpire::detail::registry::get().find_allocation(moved).has_value());
  EXPECT_EQ(rm.getAllocator(moved).getName(), named_allocator.getName());
  EXPECT_EQ(host().get_current_size(), 0u);

  for (int i = 0; i < 24; ++i) {
    EXPECT_EQ(moved_bytes[i], static_cast<unsigned char>(i + 11));
  }

  named_allocator.deallocate(moved);
}

TEST(ApiV1V2Interop, V1AllocatorSelectedReallocateWithHostPreservesV2Ownership)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");
  auto* ptr = static_cast<unsigned char*>(host().allocate(16));
  for (int i = 0; i < 16; ++i) {
    ptr[i] = static_cast<unsigned char>(0x20 + i);
  }

  auto* resized = static_cast<unsigned char*>(rm.reallocate(ptr, 64, host_allocator));

  ASSERT_NE(resized, nullptr);
  ASSERT_TRUE(umpire::detail::registry::get().find_allocation(resized).has_value());
  EXPECT_EQ(host().get_current_size(), 64u);

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(resized[i], static_cast<unsigned char>(0x20 + i));
  }

  rm.deallocate(resized);
}

TEST(ApiV1V2Interop, V1AllocatorSelectedZeroSizeReallocateWithHostReleasesV2Ownership)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");
  auto* ptr = static_cast<unsigned char*>(host().allocate(16));
  ASSERT_TRUE(umpire::detail::registry::get().find_allocation(ptr).has_value());
  EXPECT_EQ(host().get_current_size(), 16u);

  void* zero = rm.reallocate(ptr, 0, host_allocator);

  EXPECT_FALSE(umpire::detail::registry::get().find_allocation(ptr).has_value());
  EXPECT_EQ(host().get_current_size(), 0u);

  if (zero) {
    rm.deallocate(zero);
  }
}

TEST(ApiV1V2Interop, V1AllocatorSelectedZeroSizeAsyncReallocateWithHostReleasesV2Ownership)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");
  auto* ptr = static_cast<unsigned char*>(host().allocate(24));
  ASSERT_TRUE(umpire::detail::registry::get().find_allocation(ptr).has_value());
  EXPECT_EQ(host().get_current_size(), 24u);

  camp::resources::Resource ctx{camp::resources::Host{}};
  void* zero = rm.reallocate(ptr, 0, host_allocator, ctx);

  EXPECT_FALSE(umpire::detail::registry::get().find_allocation(ptr).has_value());
  EXPECT_EQ(host().get_current_size(), 0u);

  if (zero) {
    rm.deallocate(zero);
  }
}

TEST(ApiV1V2Interop, V1AllocatorSelectedReallocateRejectsDistinctAllocator)
{
  auto& rm = umpire::ResourceManager::getInstance();
  auto host_allocator = rm.getAllocator("HOST");
  auto named_allocator = rm.makeAllocator<umpire::strategy::NamedAllocationStrategy>(
      unique_allocator_name("API_V2_REALLOC_HOST"), host_allocator);

  auto* ptr = static_cast<unsigned char*>(host().allocate(16));
  ASSERT_TRUE(umpire::detail::registry::get().find_allocation(ptr).has_value());

  EXPECT_THROW(static_cast<void>(rm.reallocate(ptr, 64, named_allocator)), umpire::runtime_error);
  EXPECT_TRUE(umpire::detail::registry::get().find_allocation(ptr).has_value());
  EXPECT_EQ(host().get_current_size(), 16u);

  host().deallocate(ptr);
}
