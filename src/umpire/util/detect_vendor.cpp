//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////

#include "umpire/util/detect_vendor.hpp"

namespace umpire {

MemoryResourceTraits::vendor_type cpu_vendor_type() noexcept
{
#if defined(__x86_64__)
  return MemoryResourceTraits::vendor_type::INTEL;
#elif defined(__powerpc__)
  return MemoryResourceTraits::vendor_type::IBM;
#else
  return MemoryResourceTraits::vendor_type::UNKNOWN;
#endif
}

} // end namespace umpire
