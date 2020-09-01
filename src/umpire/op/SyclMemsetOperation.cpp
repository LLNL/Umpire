//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/SyclMemsetOperation.hpp"

#include <CL/sycl.hpp>
#include <iostream>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void SyclMemsetOperation::apply(void* src_ptr,
                                util::AllocationRecord* allocation, int value,
                                std::size_t length)
{
  cl::sycl::queue sycl_queue = allocation->strategy->getTraits().queue;
  sycl_queue.memset(src_ptr, value, length);
  sycl_queue.wait();

  UMPIRE_RECORD_STATISTIC("SyclMemsetOperation", "src_ptr",
                          reinterpret_cast<uintptr_t>(src_ptr), "value", value,
                          "size", length, "event", "memset");
}

} // end of namespace op
} // end of namespace umpire
