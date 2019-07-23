//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/RocmCopyOperation.hpp"

#include <hc.hpp>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void RocmCopyOperation::transform(
    void* src_ptr,
    void** dst_ptr,
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    std::size_t length)
{
  hc::accelerator acc;
  hc::accelerator_view av = acc.get_default_view();

  av.copy(src_ptr, *dst_ptr, length);

  av.wait();

  UMPIRE_RECORD_STATISTIC(
      "RocmCopyOperation",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "dst_ptr", reinterpret_cast<uintptr_t>(dst_ptr),
      "size", length,
      "event", "copy");
}

} // end of namespace op
} // end of namespace umpire
