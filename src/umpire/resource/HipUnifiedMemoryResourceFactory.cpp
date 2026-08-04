//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HipUnifiedMemoryResourceFactory.hpp"

#include <memory>

#include "hip/hip_runtime_api.h"
#include "umpire/resource/HipUnifiedMemoryResource.hpp"
#include "umpire/util/make_unique.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2) && !defined(UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY)
#include "umpire/resource/hip_um_memory.hpp"
#include "umpire/resource/v2_backed_resource.hpp"
#endif

namespace umpire {
namespace resource {

bool HipUnifiedMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if (name.find("UM") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> HipUnifiedMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> HipUnifiedMemoryResourceFactory::create(const std::string& name, int id,
                                                                                  MemoryResourceTraits traits)
{
#if defined(UMPIRE_V1_DELEGATE_TO_V2) && !defined(UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY)
  // Tracking=false: see the double-tracking discussion in
  // v2_backed_resource.hpp.
  //
  // Excluded when UMPIRE_ENABLE_HIP_COHERENCE_GRANULARITY is set: v1's
  // HipUnifiedMemoryResource constructs its allocator with
  // `traits.granularity` to select coarse- vs fine-grained coherence (see
  // alloc::HipMallocManagedAllocator), a knob v2's hip_um_allocator does not
  // expose. Delegating in that configuration would silently drop the
  // granularity setting, so HIP UM stays NATIVE there.
  auto v2_memory = std::make_unique<resource::hip_um_memory<resource::hip_um_allocator, false>>(name + "_v2backed");

  return util::make_unique<v2_backed_resource>(
      name, id, traits, Platform::hip, std::move(v2_memory),
      [](Platform p) { return p == Platform::hip || p == Platform::host; });
#else
  return util::make_unique<resource::HipUnifiedMemoryResource>(Platform::hip, name, id, traits);
#endif
}

MemoryResourceTraits HipUnifiedMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  hipDeviceProp_t properties;
  auto error = ::hipGetDeviceProperties(&properties, 0);

  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error, fmt::format("hipGetDeviceProperties failed with error: {}", hipGetErrorString(error)));
  }

  traits.unified = true;
  traits.size = properties.totalGlobalMem;

  traits.vendor = MemoryResourceTraits::vendor_type::amd;
  traits.kind = MemoryResourceTraits::memory_type::gddr;
  traits.used_for = MemoryResourceTraits::optimized_for::any;
  traits.resource = MemoryResourceTraits::resource_type::um;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
