//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-26, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HipDeviceResourceFactory.hpp"

#include <memory>

#include "hip/hip_runtime_api.h"
#include "umpire/resource/HipDeviceMemoryResource.hpp"
#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"

#if defined(UMPIRE_V1_DELEGATE_TO_V2)
#include "umpire/resource/hip_device_memory.hpp"
#include "umpire/resource/v2_backed_resource.hpp"
#endif

namespace umpire {
namespace resource {

bool HipDeviceResourceFactory::isValidMemoryResourceFor(const std::string& name) noexcept
{
  if ((name.find("CONST") == std::string::npos) && (name.find("DEVICE") != std::string::npos)) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource> HipDeviceResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource> HipDeviceResourceFactory::create(const std::string& name, int id,
                                                                           MemoryResourceTraits traits)
{
#if defined(UMPIRE_V1_DELEGATE_TO_V2)
  // Tracking=false: see the double-tracking discussion in
  // v2_backed_resource.hpp. Same device-restore nuance documented for the
  // CUDA device factory applies here (v1 restores the prior active device
  // after each call; v2's hip_default_allocator does not).
  auto v2_memory = std::make_unique<resource::hip_device_memory<resource::hip_default_allocator, false>>(
      name + "_v2backed", traits.id, resource::hip_default_allocator(traits.id));

  return util::make_unique<v2_backed_resource>(
      name, id, traits, Platform::hip, std::move(v2_memory),
      [](Platform p) { return p == Platform::hip || p == Platform::host; });
#else
  return util::make_unique<resource::HipDeviceMemoryResource>(Platform::hip, name, id, traits);
#endif
}

MemoryResourceTraits HipDeviceResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;

  hipDeviceProp_t properties;
  auto error = ::hipGetDeviceProperties(&properties, 0);

  if (error != hipSuccess) {
    UMPIRE_ERROR(runtime_error, fmt::format("hipGetDeviceProperties failed with error: {}", hipGetErrorString(error)));
  }

  traits.unified = false;
  traits.size = properties.totalGlobalMem;

  traits.vendor = MemoryResourceTraits::vendor_type::amd;
  traits.kind = MemoryResourceTraits::memory_type::gddr;
  traits.used_for = MemoryResourceTraits::optimized_for::any;
  traits.resource = MemoryResourceTraits::resource_type::device;

  return traits;
}

} // end of namespace resource
} // end of namespace umpire
