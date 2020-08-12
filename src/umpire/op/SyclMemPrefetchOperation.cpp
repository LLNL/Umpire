//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/SyclMemPrefetchOperation.hpp"

#include <CL/sycl.hpp>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void SyclMemPrefetchOperation::apply(void* src_ptr,
                                     util::AllocationRecord* allocation,
                                     int value, std::size_t length)
{
  if (allocation->strategy->getTraits().id != value) {
    UMPIRE_ERROR("SYCL memPrefetch failed with invalid deviceID  = " << value);
  }

  cl::sycl::queue sycl_queue(allocation->strategy->getTraits().queue);
  auto sycl_device = sycl_queue.get_device();
  auto ctxt = sycl_queue.get_context();

  cl::sycl::usm::alloc src_ptr_kind = get_pointer_type(src_ptr, ctxt);

  if (sycl_device.get_info<cl::sycl::info::device::usm_shared_allocations>() &&
      cl::sycl::usm::alloc::shared == src_ptr_kind) {
    sycl_queue.prefetch(src_ptr, length);
    sycl_queue.wait();
  } else {
    UMPIRE_ERROR("SYCL memPrefetch failed ( bytes = " << length << " )");
  }
}

} // end of namespace op
} // end of namespace umpire
