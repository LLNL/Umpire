//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/SyclMemsetOperation.hpp"

#include <iostream>

#include "umpire/strategy/AllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/sycl_compat.hpp"

namespace umpire {
namespace op {

void SyclMemsetOperation::apply(void* src_ptr, util::AllocationRecord* allocation, int value, std::size_t length)
{
  auto sycl_queue = allocation->strategy->getTraits().queue;
  sycl_queue->memset(src_ptr, value, length);
  sycl_queue->wait();
}

} // end of namespace op
} // end of namespace umpire
