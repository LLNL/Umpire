//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/SyclCopyOperation.hpp"

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/sycl_compat.hpp"

namespace umpire {
namespace op {

void SyclCopyOperation::transform(void* src_ptr, void** dst_ptr,
                                  umpire::util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
                                  umpire::util::AllocationRecord* dst_allocation, std::size_t length)
{
  auto sycl_queue = dst_allocation->strategy->getTraits().queue;

  sycl_queue->memcpy(*dst_ptr, src_ptr, length);
  sycl_queue->wait();
}

} // end of namespace op
} // end of namespace umpire
