//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/GenericReallocateOperation.hpp"

#include <cstdlib>

#include "umpire/ResourceManager.hpp"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/AllocationRecord.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void GenericReallocateOperation::transform(
    void* current_ptr, void** new_ptr,
    util::AllocationRecord* current_allocation,
    util::AllocationRecord* new_allocation, std::size_t new_size)
{
  auto allocator = new_allocation->strategy;
  *new_ptr = allocator->allocate(new_size);

  const std::size_t old_size = current_allocation->size;
  const std::size_t copy_size = (old_size > new_size) ? new_size : old_size;

  ResourceManager::getInstance().copy(*new_ptr, current_ptr, copy_size);

  UMPIRE_RECORD_STATISTIC("GenericReallocate", "current_ptr",
                          reinterpret_cast<uintptr_t>(current_ptr), "new_ptr",
                          reinterpret_cast<uintptr_t>(*new_ptr), "new_size",
                          new_size, "event", "reallocate");

  allocator->deallocate(current_ptr);
}

} // end of namespace op
} // end of namespace umpire
