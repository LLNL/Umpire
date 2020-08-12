//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/HostMemsetOperation.hpp"

#include <cstring>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void HostMemsetOperation::apply(
    void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(allocation),
    int value, std::size_t length)
{
  std::memset(src_ptr, value, length);

  UMPIRE_RECORD_STATISTIC("HostMemsetOperation", "src_ptr",
                          reinterpret_cast<uintptr_t>(src_ptr), "value", value,
                          "size", length, "event", "memset");
}

} // end of namespace op
} // end of namespace umpire
