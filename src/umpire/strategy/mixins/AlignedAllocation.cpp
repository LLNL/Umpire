//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/strategy/mixins/AlignedAllocation.hpp"

namespace umpire {
namespace strategy {
namespace mixins {

AlignedAllocation::AlignedAllocation(
  std::size_t alignment,
  strategy::AllocationStrategy* strategy)
    : m_allocator{ strategy },
      m_alignment{ alignment },
      m_mask{ static_cast<uintptr_t>( ~(m_alignment-1)) }
{
}

} // namespace mixins
} // namespace strategy
} // namespace umpire
