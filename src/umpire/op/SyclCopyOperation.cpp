//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/SyclCopyOperation.hpp"

#include <CL/sycl.hpp>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void SyclCopyOperation::transform(
    void* src_ptr, void** dst_ptr,
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    umpire::util::AllocationRecord* dst_allocation, std::size_t length)
{
  cl::sycl::queue sycl_queue(dst_allocation->strategy->getTraits().queue);
  auto ctxt = sycl_queue.get_context();

  sycl_queue.memcpy(*dst_ptr, src_ptr, length);
  sycl_queue.wait();

  UMPIRE_RECORD_STATISTIC("SyclCopyOperation", "src_ptr",
                          reinterpret_cast<uintptr_t>(src_ptr), "dst_ptr",
                          reinterpret_cast<uintptr_t>(*dst_ptr), "size", length,
                          "event", "copy");
}

} // end of namespace op
} // end of namespace umpire
