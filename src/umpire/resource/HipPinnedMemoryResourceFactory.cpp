//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HipPinnedMemoryResourceFactory.hpp"

#include <memory>

#include "umpire/util/make_unique.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2) && !defined(UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY)
#include "umpire/resource/hip_pinned_memory.hpp"
#include "umpire/resource/v2_backed_resource.hpp"
#endif

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
#if defined(UMPIRE_V1_DELEGATE_TO_V2) && !defined(UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY)
  // Tracking=false: see the double-tracking discussion in
  // v2_backed_resource.hpp. Excluded under
  // UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY for the same reason documented
  // in HipUnifiedMemoryResourceFactory.cpp (v2's hip_pinned_allocator has no
  // granularity knob).
  auto v2_memory =
      std::make_unique<resource::hip_pinned_memory<resource::hip_pinned_allocator, false>>(name + "_v2backed");

  return util::make_unique<v2_backed_resource>(
      name, id, traits, Platform::hip, std::move(v2_memory),
      [](Platform p) { return p == Platform::hip || p == Platform::host; });
#else
  return util::make_unique<resource::HipPinnedMemoryResource>(Platform::hip, name, id, traits);
#endif
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
