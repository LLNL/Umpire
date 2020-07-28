//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/util/AlignedAllocation.hpp"

namespace umpire {
namespace util {
AlignedAllocation::AlignedAllocation(std::size_t alignment)
    : m_alignment{ alignment },
      m_mask{ static_cast<uintptr_t>( ~(m_alignment-1)) }
{
}
} // namespace umpire
} // namespace util
