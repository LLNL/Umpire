//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/FileMemoryResourceFactory.hpp"

#include "umpire/resource/FileMemoryResource.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool FileMemoryResourceFactory::isValidMemoryResourceFor(
    const std::string& name) noexcept
{
  if (name.find("FILE") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> FileMemoryResourceFactory::create(
    const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> FileMemoryResourceFactory::create(
    const std::string& name, int id, MemoryResourceTraits traits)
{
  return util::make_unique<FileMemoryResource>(Platform::undefined, name, id,
                                               traits);
}

MemoryResourceTraits FileMemoryResourceFactory::getDefaultTraits()
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
