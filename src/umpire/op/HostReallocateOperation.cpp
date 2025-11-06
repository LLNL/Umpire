//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-25, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/HostReallocateOperation.hpp"

#include <cstdlib>

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/config.hpp"
#include "umpire/strategy/mixins/Inspector.hpp"
#include "umpire/util/error.hpp"

#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
#include "umpire/util/allocation_metadata.hpp"
#endif

namespace umpire {
namespace op {

void HostReallocateOperation::transform(void* current_ptr, void** new_ptr, util::AllocationRecord* current_allocation,
                                        util::AllocationRecord* new_allocation, std::size_t new_size)
{
  auto allocator = umpire::Allocator(new_allocation->strategy);
  const std::size_t old_size = current_allocation->size;

  //
  // Since Umpire implements its own semantics for zero-length allocations, we
  // cannot simply call ::realloc() with a pointer to a zero-length allocation.
  //
  if (old_size == 0) {
    *new_ptr = allocator.allocate(new_size);
    const std::size_t copy_size = (old_size > new_size) ? new_size : old_size;

    ResourceManager::getInstance().copy(*new_ptr, current_ptr, copy_size);
    allocator.deallocate(current_ptr);
  } else {
#ifdef UMPIRE_ENABLE_HEADER_INTROSPECTION
    // Header mode - destruct old header, realloc with new header space, construct new header
    auto [old_record, old_base_ptr] = util::destruct_allocation<util::AllocationRecord>(current_ptr);

    // Realloc with space for new header
    std::size_t new_total_size = util::allocation_size(new_size);
    void* new_base_ptr = ::realloc(old_base_ptr, new_total_size);

    if (!new_base_ptr) {
      UMPIRE_ERROR(runtime_error, fmt::format("::realloc(current_ptr={}, old_size={}, new_size={}) failed.",
                                              current_ptr, old_size, new_size));
    }

    // Construct new header and get user pointer
    *new_ptr = util::construct_allocation<util::AllocationRecord>(
      new_base_ptr,
      util::allocation_data(new_base_ptr),
      new_size,
      new_allocation->strategy,
      old_record.name
    );
#else
    // Original map-based path
    auto old_record = ResourceManager::getInstance().deregisterAllocation(current_ptr);
    *new_ptr = ::realloc(current_ptr, new_size);

    if (!*new_ptr) {
      UMPIRE_ERROR(runtime_error, fmt::format("::realloc(current_ptr={}, old_size={}, new_size={}) failed.",
                                              current_ptr, old_record.size, new_size));
    }

    ResourceManager::getInstance().registerAllocation(*new_ptr, {*new_ptr, new_size, new_allocation->strategy});
#endif
  }
}

} // end of namespace op
} // end of namespace umpire
