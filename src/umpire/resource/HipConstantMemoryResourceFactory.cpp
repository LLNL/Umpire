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
#include "umpire/resource/HipConstantMemoryResourceFactory.hpp"

#include "umpire/resource/HipConstantMemoryResource.hpp"

#include "umpire/util/Macros.hpp"

namespace umpire {
namespace resource {

bool
HipConstantMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name)
  noexcept
{
  if (name.compare("DEVICE_CONST") == 0) {
    return true;
  } else {
    return false;
  }
}

resource::MemoryResource*
HipConstantMemoryResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 64*1024;

  traits.vendor = MemoryResourceTraits::vendor_type::AMD;
  traits.kind = MemoryResourceTraits::memory_type::GDDR;

  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return new resource::HipConstantMemoryResource("DEVICE_CONST", id, traits);
}

} // end of namespace resource
} // end of namespace umpire
