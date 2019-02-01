//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////

#include "umpire/resource/DetectVendor.hpp"

namespace umpire {

resource::MemoryResourceTraits::vendor_type cpu_vendor_type() noexcept {
#if defined(__x86_64__)
  return resource::MemoryResourceTraits::vendor_type::INTEL;
#elif defined(__powerpc__)
  return resource::MemoryResourceTraits::vendor_type::IBM;
#else
  return resource::MemoryResourceTraits::vendor_type::UNKNOWN;
#endif
}

}
