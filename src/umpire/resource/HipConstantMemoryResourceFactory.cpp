//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HipConstantMemoryResourceFactory.hpp"

#include "umpire/resource/HipConstantMemoryResource.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool HipConstantMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if (name.compare("DEVICE_CONST") == 0) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> HipConstantMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> HipConstantMemoryResourceFactory::create(const std::string& name, int id,
                                                                                   MemoryResourceTraits traits)
{
  return util::make_unique<resource::HipConstantMemoryResource>(name, id, traits);
}

MemoryResourceTraits HipConstantMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 64 * 1024;

  traits.vendor = MemoryResourceTraits::vendor_type::amd;
  traits.kind = MemoryResourceTraits::memory_type::gddr;
  traits.used_for = MemoryResourceTraits::optimized_for::any;
  traits.resource = MemoryResourceTraits::resource_type::device_const;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
