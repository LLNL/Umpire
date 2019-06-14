//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018-2019, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory
//
// Created by David Beckingsale, david@llnl.gov
// LLNL-CODE-747640
//
// All rights reserved.
//
// This file is part of Umpire.
//
// For details, see https://github.com/LLNL/Umpire
// Please also see the LICENSE file for MIT license.
//////////////////////////////////////////////////////////////////////////////
#include "umpire/resource/HipDeviceResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"
#include "umpire/alloc/HipMallocAllocator.hpp"

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

resource::MemoryResource*
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

  return new resource::DefaultMemoryResource<alloc::HipMallocAllocator>(Platform::hip, "DEVICE", id, traits);
}

} // end of namespace resource
} // end of namespace umpire
