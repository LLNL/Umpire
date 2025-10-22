//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/util/HeaderIntrospection.hpp"

#include "umpire/config.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/Platform.hpp"
#include "umpire/util/backtrace.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace util {

void* insertHeader(void* base_ptr, std::size_t size, strategy::AllocationStrategy* strategy, const std::string& name)
{
  UMPIRE_LOG(Debug, "insertHeader(" << base_ptr << ", " << size << ", " << strategy->getName() << ")");

  // Allocate AllocationRecord from pool
  void* record_mem;
  {
    std::lock_guard<std::mutex> lock(detail::getRecordPoolMutex());
    record_mem = detail::getRecordPool().allocate();
  }

  // Calculate user pointer (after header)
  uintptr_t base = reinterpret_cast<uintptr_t>(base_ptr);
  void* user_ptr = reinterpret_cast<void*>(base + sizeof(IntrospectionHeader));

  // Construct AllocationRecord in place
  AllocationRecord* record = new (record_mem) AllocationRecord{user_ptr, size, strategy, name};

#if defined(UMPIRE_ENABLE_BACKTRACE)
  // Record backtrace if enabled
  record->allocation_backtrace = umpire::util::backtracer<umpire::util::trace_always>::get_backtrace();
#endif

  // Write header
  IntrospectionHeader* header = reinterpret_cast<IntrospectionHeader*>(base);
  header->record = record;

  UMPIRE_LOG(Debug, "Inserted header at " << base << ", user ptr: " << user_ptr << ", record: " << record);

  return user_ptr;
}

AllocationRecord* getRecord(void* user_ptr)
{
  UMPIRE_LOG(Debug, "getRecord(" << user_ptr << ")");

  uintptr_t base = reinterpret_cast<uintptr_t>(user_ptr) - sizeof(IntrospectionHeader);
  IntrospectionHeader* header = reinterpret_cast<IntrospectionHeader*>(base);

  UMPIRE_LOG(Debug, "Retrieved record " << header->record << " from header at " << reinterpret_cast<void*>(base));

  return header->record;
}

std::pair<AllocationRecord, void*> removeHeader(void* user_ptr)
{
  UMPIRE_LOG(Debug, "removeHeader(" << user_ptr << ")");

  // Read header
  uintptr_t base = reinterpret_cast<uintptr_t>(user_ptr) - sizeof(IntrospectionHeader);
  IntrospectionHeader* header = reinterpret_cast<IntrospectionHeader*>(base);
  AllocationRecord* record = header->record;

  // Copy record data before freeing
  AllocationRecord record_copy = *record;

  // Destroy and free record back to pool
  {
    std::lock_guard<std::mutex> lock(detail::getRecordPoolMutex());
    record->~AllocationRecord();
    detail::getRecordPool().deallocate(record);
  }

  void* base_ptr = reinterpret_cast<void*>(base);

  UMPIRE_LOG(Debug, "Removed header at " << base_ptr << ", freed record " << record);

  return {record_copy, base_ptr};
}

bool supportsHeaderIntrospection(strategy::AllocationStrategy* strategy)
{
  // Get platform and traits from strategy
  Platform platform = strategy->getPlatform();
  MemoryResourceTraits traits = strategy->getTraits();

  UMPIRE_LOG(Debug, "Checking header support for " << strategy->getName() << ", platform: " << platform_to_string(platform)
                                                    << ", unified: " << traits.unified);

  // Header introspection only works with host-accessible memory
  switch (platform) {
    case Platform::host:
      // Host memory always supports headers
      return true;

#ifdef UMPIRE_ENABLE_CUDA
    case Platform::cuda:
      // CUDA unified/managed memory supports direct host access
      return traits.unified;
#endif

#ifdef UMPIRE_ENABLE_HIP
    case Platform::hip:
      // HIP managed memory supports direct host access
      return traits.unified;
#endif

#ifdef UMPIRE_ENABLE_SYCL
    case Platform::sycl:
      // SYCL USM shared memory supports direct host access
      // Check if resource type is shared (not device-only)
      return (traits.resource == MemoryResourceTraits::resource_type::shared);
#endif

    default:
      // Other platforms (omp_target, etc.) default to no header support
      return false;
  }
}

} // end of namespace util
} // end of namespace umpire
