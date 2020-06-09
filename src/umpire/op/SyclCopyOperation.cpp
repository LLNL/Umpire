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
    void* src_ptr,
    void** dst_ptr,
    umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
    umpire::util::AllocationRecord* dst_allocation,
    std::size_t length)
{
  cl::sycl::device sycl_device(dst_allocation->strategy->getTraits().deviceID);
  cl::sycl::queue sycl_queue(sycl_device);
  auto ctxt = sycl_queue.get_context();

  // get device info for the pointers
  cl::sycl::device src_dev = get_pointer_device(src_ptr, ctxt);
  cl::sycl::device dst_dev = get_pointer_device(*dst_ptr, ctxt);

  // copy within the same device
  if (src_dev == dst_dev) {
      sycl_queue.memcpy(*dst_ptr, src_ptr, length);
      sycl_queue.wait();
  }
  else
  {
      UMPIRE_ERROR("SYCL deviceTodevice memcpy failed( bytes = " << length << " )");
  }

  UMPIRE_RECORD_STATISTIC(
      "SyclCopyOperation",
      "src_ptr", reinterpret_cast<uintptr_t>(src_ptr),
      "dst_ptr", reinterpret_cast<uintptr_t>(*dst_ptr),
      "size", length,
      "event", "copy");
}

} // end of namespace op
} // end of namespace umpire
