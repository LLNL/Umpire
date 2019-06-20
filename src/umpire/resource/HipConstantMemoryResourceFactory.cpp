//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
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
