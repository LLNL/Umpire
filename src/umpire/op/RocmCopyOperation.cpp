//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
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
    size_t length)
{
  hc::accelerator acc;
  hc::accelerator_view av = acc.get_default_view();

  av.copy(src_ptr, *dst_ptr, length);

  UMPIRE_RECORD_STATISTIC(
      "RocmCopyOperation",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "dst_ptr", reinterpret_cast<uintptr_t>(dst_ptr),
      "size", length,
      "event", "copy");
}

} // end of namespace op
} // end of namespace umpire
