//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HipConstantMemoryResourceFactory.hpp"

#include <memory>

#include "umpire/resource/HipConstantMemoryResource.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/resource/hip_device_const_memory.hpp"
#include "umpire/resource/v2_backed_resource.hpp"
#endif

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
#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  // Tracking=false: see the double-tracking discussion in
  // v2_backed_resource.hpp. All named hip_device_const_memory instances
  // share the same fixed 64KB __constant__ buffer, matching v1 semantics.
  auto v2_memory =
      std::make_unique<resource::hip_device_const_memory<resource::hip_device_const_allocator, false>>(
          name + "_v2backed");

  return util::make_unique<v2_backed_resource>(name, id, traits, Platform::hip, std::move(v2_memory),
                                                [](Platform p) { return p == Platform::hip; });
#else
  return util::make_unique<resource::HipConstantMemoryResource>(name, id, traits);
#endif
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
