//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
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
    void* src_ptr,
    void** dst_ptr,
    util::AllocationRecord *src_allocation,
    util::AllocationRecord *dst_allocation,
    std::size_t length)
{
  auto allocator = dst_allocation->strategy;
  *dst_ptr = allocator->allocate(length);

  std::size_t old_size = src_allocation->size;
  std::size_t copy_size = ( old_size > length ) ? length : old_size;

  ResourceManager::getInstance().copy(*dst_ptr, src_ptr, copy_size);

  UMPIRE_RECORD_STATISTIC(
      "GenericReallocate",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "dst_ptr", reinterpret_cast<uintptr_t>(*dst_ptr),
      "size", length,
      "event", "reallocate");

  allocator->deallocate(src_ptr);
}

} // end of namespace op
} // end of namespace umpire
