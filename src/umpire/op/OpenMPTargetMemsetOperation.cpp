//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/OpenMPTargetMemsetOperation.hpp"

#include <cstring>

#include "omp.h"
#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void OpenMPTargetMemsetOperation::apply(void* src_ptr,
                                        util::AllocationRecord* src_allocation,
                                        int val, std::size_t length)
{
  int device = src_allocation->strategy->getTraits().id;
  unsigned char* data_ptr{static_cast<unsigned char*>(src_ptr)};

#pragma omp target is_device_ptr(data_ptr) device(device)
#pragma omp teams distribute parallel for schedule(static, 1)
  for (std::size_t i = 0; i < length; ++i) {
    data_ptr[i] = static_cast<unsigned char>(val);
  }

  UMPIRE_RECORD_STATISTIC("OpenMPTargetMemsetOperation", "src_ptr",
                          reinterpret_cast<uintptr_t>(src_ptr), "value", value,
                          "size", length, "event", "memset");
}

} // end of namespace op
} // end of namespace umpire
