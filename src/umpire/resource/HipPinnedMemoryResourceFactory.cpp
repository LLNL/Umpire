//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
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
#include "umpire/resource/HipPinnedMemoryResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"

#include "umpire/alloc/HipPinnedAllocator.hpp"

namespace umpire {
namespace resource {

bool
HipPinnedMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name)
  noexcept
{
  if (name.compare("PINNED") == 0) {
    return true;
  } else {
    return false;
  }
}

resource::MemoryResource*
HipPinnedMemoryResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 0; // size of system memory?

  traits.vendor = MemoryResourceTraits::vendor_type::AMD;
  traits.kind = MemoryResourceTraits::memory_type::DDR;
  traits.used_for = MemoryResourceTraits::optimized_for::access;

  return new resource::DefaultMemoryResource<alloc::HipPinnedAllocator>(Platform::hip, "PINNED", id, traits);
}

} // end of namespace resource
} // end of namespace umpire
