//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/SyclCopyFromOperation.hpp"

#include <CL/sycl.hpp>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void SyclCopyFromOperation::transform(
    void* src_ptr, void** dst_ptr, util::AllocationRecord* src_allocation,
    util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation),
    std::size_t length)
{
  auto sycl_queue = src_allocation->strategy->getTraits().queue;
  sycl_queue->memcpy(*dst_ptr, src_ptr, length);
  sycl_queue->wait();
}

} // end of namespace op
} // end of namespace umpire
