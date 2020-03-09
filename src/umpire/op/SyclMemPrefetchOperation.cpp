//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/SyclMemPrefetchOperation.hpp"

#include <CL/sycl.hpp>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void
SyclMemPrefetchOperation::apply(
    void* src_ptr,
    util::AllocationRecord*  UMPIRE_UNUSED_ARG(allocation),
    int value, 
    std::size_t length)
{
  int device{value}; // todo: not being used for now

  cl::sycl::device sycl_device(allocation->strategy->getTraits().deviceID);
  cl::sycl::queue sycl_queue(sycl_device);

  if (sycl_device.get_info<cl::sycl::info::device::host_unified_memory() &&
      sycl_device.get_info<cl::sycl::info::device::usm_shared_allocations>() &&
      cl::sycl::usm::alloc::shared == get_pointer_type(src_ptr, sycl_device.get_context())) {
    sycl_queue.submit([&](handler &cgh)
    {
      cgh.prefetch(src_ptr, length);
    });
    sycl_queue.wait_and_throw();
  }
}

} // end of namespace op
} // end of namespace umpire
