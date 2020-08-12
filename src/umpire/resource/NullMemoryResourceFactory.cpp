//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/NullMemoryResourceFactory.hpp"

#include "umpire/resource/NullMemoryResource.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool NullMemoryResourceFactory::isValidMemoryResourceFor(
    const std::string& name) noexcept
{
  if (name.compare("__umpire_internal_null") == 0) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> NullMemoryResourceFactory::create(
    const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  return create("__umpire_internal_null", id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> NullMemoryResourceFactory::create(
    const std::string& name, int id, MemoryResourceTraits traits)
{
  return util::make_unique<NullMemoryResource>(Platform::undefined, name, id,
                                               traits);
}

MemoryResourceTraits NullMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 0;

  traits.vendor = MemoryResourceTraits::vendor_type::UNKNOWN;
  traits.kind = MemoryResourceTraits::memory_type::UNKNOWN;
  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
