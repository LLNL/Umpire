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
#include "umpire/resource/SICMResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"

#include "umpire/alloc/SICMAllocator.hpp"

#include "umpire/util/detect_vendor.hpp"

namespace umpire {
namespace resource {

bool
SICMResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if (name.compare("SICM") == 0) {
    return true;
  } else {
    return false;
  }
}

resource::MemoryResource*
SICMResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 0;

  traits.vendor = cpu_vendor_type();
  traits.kind = MemoryResourceTraits::memory_type::UNKNOWN;
  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return new DefaultMemoryResource<alloc::SICMAllocator>(Platform::cpu, "SICM", id, traits);
}

} // end of namespace resource
} // end of namespace umpire
