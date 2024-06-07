//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#ifndef UMPIRE_HipAllocator_HPP
#define UMPIRE_HipAllocator_HPP

#include "umpire/util/Macros.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"
#include "umpire/util/error.hpp"

namespace umpire {
namespace alloc {

struct HipAllocator {
  HipAllocator() : m_granularity{MemoryResourceTraits::granularity_type::unknown}
  {
  }

  HipAllocator(MemoryResourceTraits::granularity_type g) : m_granularity{g}
  {
  }

  MemoryResourceTraits::granularity_type m_granularity;
};

} // end of namespace alloc
} // end of namespace umpire

#endif // UMPIRE_HipAllocator_HPP
