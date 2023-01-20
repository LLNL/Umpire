//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/op/HostCopyOperation.hpp"

#include <cstring>

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace op {

void HostCopyOperation::transform(void* src_ptr, void** dst_ptr,
                                  util::AllocationRecord* UMPIRE_UNUSED_ARG(src_allocation),
                                  util::AllocationRecord* UMPIRE_UNUSED_ARG(dst_allocation), std::size_t length)
{
  std::memcpy(*dst_ptr, src_ptr, length);
}

} // end of namespace op
} // end of namespace umpire
