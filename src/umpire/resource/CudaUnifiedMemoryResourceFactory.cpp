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
#include "umpire/resource/CudaUnifiedMemoryResourceFactory.hpp"

#include "umpire/resource/DefaultMemoryResource.hpp"

#include "umpire/alloc/CudaMallocManagedAllocator.hpp"

#include <cuda_runtime_api.h>

namespace umpire {
namespace resource {

bool
CudaUnifiedMemoryResourceFactory::isValidMemoryResourceFor(const std::string& name,
                                                           const MemoryResourceTraits UMPIRE_UNUSED_ARG(traits))
  noexcept
{
  if (name.compare("UM") == 0) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<MemoryResource>
CudaUnifiedMemoryResourceFactory::create(const std::string& UMPIRE_UNUSED_ARG(name), int id)
{
  MemoryResourceTraits traits;

  cudaDeviceProp properties;
  auto error = ::cudaGetDeviceProperties(&properties, 0);

  if (error != cudaSuccess) {
    UMPIRE_ERROR("cudaGetDeviceProperties failed with error: " << cudaGetErrorString(error));
  }

  traits.unified = true;
  traits.size = properties.totalGlobalMem; // plus system size?

  traits.vendor = MemoryResourceTraits::vendor_type::NVIDIA;
  traits.kind = MemoryResourceTraits::memory_type::GDDR;
  traits.used_for = MemoryResourceTraits::optimized_for::any;

  return std::make_shared<resource::DefaultMemoryResource<alloc::CudaMallocManagedAllocator> >(Platform::cuda, "UM", id, traits);
}

} // end of namespace resource
} // end of namespace umpire
