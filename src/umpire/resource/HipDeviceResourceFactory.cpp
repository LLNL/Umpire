//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC and Umpire
// project contributors. See the COPYRIGHT file for details.
//
// SPDX-License-Identifier: (MIT)
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HipDeviceResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/alloc/HipMallocAllocator.hpp"

#include "umpire/util/Macros.hpp"
#include "umpire/util/make_unique.hpp"

#include <hip/hip_runtime.h>

namespace umpire {
namespace resource {

bool
HipDeviceResourceFactory::isValidMemoryResourceFor(const std::string& name)
  noexcept
{
  if (name.compare("DEVICE") == 0) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<resource::MemoryResource>
HipDeviceResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  MemoryResourceTraits traits;

  hipDeviceProp_t properties;
  auto error = ::hipGetDeviceProperties(&properties, 0);

  if (error != hipSuccess) {
    UMPIRE_ERROR("hipGetDeviceProperties failed with error: " << hipGetErrorString(error));
  }

  traits.unified = false;
  traits.size = properties.totalGlobalMem;

  traits.vendor = MemoryResourceTraits::vendor_type::AMD;
  traits.kind = MemoryResourceTraits::memory_type::GDDR;
  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return util::make_unique<resource::DefaultMemoryResource<alloc::HipMallocAllocator>>(Platform::hip, "DEVICE", id, traits);
}

} // end of namespace resource
} // end of namespace umpire
