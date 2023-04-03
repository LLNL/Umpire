//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HipPinnedMemoryResourceFactory.hpp"

#include "umpire/alloc/HipPinnedAllocator.hpp"
#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool HipPinnedMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if (name.find("PINNED") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> HipPinnedMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> HipPinnedMemoryResourceFactory::create(const std::string& name, int id,
                                                                                 MemoryResourceTraits traits)
{
  return util::make_unique<resource::DefaultMemoryResource<alloc::HipPinnedAllocator>>(Platform::hip, name, id, traits);
}

MemoryResourceTraits HipPinnedMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 0; // size of system memory?

  traits.vendor = MemoryResourceTraits::vendor_type::amd;
  traits.kind = MemoryResourceTraits::memory_type::ddr;
  traits.used_for = MemoryResourceTraits::optimized_for::access;
  traits.resource = MemoryResourceTraits::resource_type::pinned;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
