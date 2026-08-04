//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HostSharedMemoryResourceFactory.hpp"

#include "umpire/resource/HostSharedMemoryResource.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/error.hpp"
#include "umpire/util/make_unique.hpp"
#include "umpire/util/shared_memory_helper.hpp"

namespace umpire {
namespace resource {

bool HostSharedMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  return umpire::util::matchesSharedMemoryResource(name, "POSIX");
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
  // NOTE (UMPIRE_V1_DELEGATE_TO_V2): intentionally left NATIVE. v1's
  // HostSharedMemoryResource exposes extra virtuals (allocate_named(),
  // find_pointer_from_name()) that v2_backed_resource's generic
  // allocate(bytes)/deallocate(ptr) adapter cannot forward, and v2's
  // shared_memory has a differing anonymous-allocate semantic (it
  // synthesizes a name rather than throwing as v1 does). Reconciling those
  // API and behavioral differences is out of scope for the generic adapter.
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
