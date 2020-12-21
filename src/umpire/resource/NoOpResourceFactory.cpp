//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/NoOpResourceFactory.hpp"
#include "umpire/resource/NoOpMemoryResource.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"
#include "umpire/util/detect_vendor.hpp"

namespace umpire {
namespace resource {

bool NoOpResourceFactory::isValidMemoryResourceFor(
    const std::string& name) noexcept
{
  if (name.find("NO_OP") != std::string::npos)
    return true;
  else 
    return false;
  
}

std::unique_ptr<resource::MemoryResource> NoOpResourceFactory::create(
    const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> NoOpResourceFactory::create(
    const std::string& name, int id, MemoryResourceTraits traits)
{
  return util::make_unique<resource::NoOpMemoryResource>(
      Platform::host, name, id, traits);
}

MemoryResourceTraits NoOpResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 0;

  traits.vendor = MemoryResourceTraits::vendor_type::unknown;
  traits.kind = MemoryResourceTraits::memory_type::unknown;
  traits.used_for = MemoryResourceTraits::optimized_for::any;
  traits.resource = MemoryResourceTraits::resource_type::unknown;

  traits.id = 0;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
