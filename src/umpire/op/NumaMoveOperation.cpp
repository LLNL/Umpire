//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/NumaMoveOperation.hpp"

#include <cstring>
#include <memory>

#include "umpire/util/Macros.hpp"
#include "umpire/util/numa.hpp"

#include "umpire/strategy/NumaPolicy.hpp"

namespace umpire {
namespace op {

void NumaMoveOperation::transform(
    void* src_ptr,
    void** dst_ptr,
    util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    util::AllocationRecord* dst_allocation,
    std::size_t length)
{
  auto numa_allocator = static_cast<strategy::NumaPolicy*>(dst_allocation->m_strategy);

  *dst_ptr = src_ptr;
  numa::move_to_node(*dst_ptr, length, numa_allocator->getNode());

  UMPIRE_RECORD_STATISTIC(
      "NumaMoveOperation",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "dst_ptr", reinterpret_cast<uintptr_t>(dst_ptr),
      "size", length,
      "event", "move");
}

} // end of namespace op
} // end of namespace umpire
