//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HipUnifiedMemoryResourceFactory.hpp"

#include "hip/hip_runtime_api.h"

#include "umpire/alloc/HipMallocManagedAllocator.hpp"
#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/util/make_unique.hpp"

namespace umpire {
namespace resource {

bool HipUnifiedMemoryResourceFactory::isValidMemoryResourceFor(
    const std::string& name) noexcept
{
  if (name.find("UM") != std::string::npos) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource>
HipUnifiedMemoryResourceFactory::create(const std::string& name, int id)
{
  return create(name, id, getDefaultTraits());
}

std::unique_ptr<resource::MemoryResource>
HipUnifiedMemoryResourceFactory::create(const std::string& name, int id,
                                         MemoryResourceTraits traits)
{
  return util::make_unique<
      resource::DefaultMemoryResource<alloc::HipMallocManagedAllocator>>(
      Platform::hip, name, id, traits);
}

MemoryResourceTraits HipUnifiedMemoryResourceFactory::getDefaultTraits()
{
  MemoryResourceTraits traits;
  
  hipDeviceProp_t properties;
  auto error = ::hipGetDeviceProperties(&properties, 0);

  if (error != hipSuccess) {
    UMPIRE_ERROR("hipGetDeviceProperties failed with error: "
                 << hipGetErrorString(error));
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
