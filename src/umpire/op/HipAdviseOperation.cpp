//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/op/HipAdviseOperation.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace op {

HipAdviseOperation::HipAdviseOperation(hipMemoryAdvise a) : m_advise(a)
{
}

void HipAdviseOperation::apply(void* src_ptr, util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation), int val,
                               std::size_t length)
{
  int device = val;
  auto error = ::hipMemAdvise(src_ptr, length, m_advise, device);

  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error,
                 fmt::format("hipMemAdvise( src_ptr = {}, length = {}, device = {}) failed with error: {}", src_ptr,
                             length, device, hipGetErrorString(error)));
  }
}

} // end of namespace op
} // end of namespace umpire
