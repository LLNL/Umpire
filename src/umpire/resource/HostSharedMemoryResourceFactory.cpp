//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HostSharedMemoryResourceFactory.hpp"

#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool HostSharedMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if (name.find("SHARED") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> HostSharedMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> HostSharedMemoryResourceFactory::create(const std::string& name, int id,
                                                                                  MemoryResourceTraits traits)
{
  if (traits.scope != MemoryResourceTraits::shared_scope::node) {
    UMPIRE_ERROR(runtime_error, "HostSharedMemoryResource only supports shared_scope::node");
  }
  return util::make_unique<HostSharedMemoryResource>(Platform::host, name, id, traits);
}

MemoryResourceTraits HostSharedMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  traits.unified = false;
  traits.size = 0;

  traits.vendor = MemoryResourceTraits::vendor_type::unknown;
  traits.kind = MemoryResourceTraits::memory_type::unknown;
  traits.used_for = MemoryResourceTraits::optimized_for::any;
  traits.resource = MemoryResourceTraits::resource_type::shared;
  traits.size = 16 * 1024 * 1024;
  traits.scope = MemoryResourceTraits::shared_scope::node;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
